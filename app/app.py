import os
import re
import json
import time
import logging
import requests
from collections import defaultdict
from flask import Flask, render_template, request, jsonify
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
import config

TRANSCRIPT_DIR = "/var/log/rag-lab"
TRANSCRIPT_PATH = os.path.join(TRANSCRIPT_DIR, "transcript.jsonl")
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)


def log_interaction(query: str, response: dict):
    """Silently records every request/response pair for later evaluation.
    Never allowed to break the actual request if logging fails."""
    try:
        with open(TRANSCRIPT_PATH, "a") as f:
            f.write(json.dumps({
                "ts": time.time(),
                "query": query,
                "response": response,
            }) + "\n")
    except Exception:
        pass


app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag")

_SESSIONS = defaultdict(list)  # session_id -> list of {"query","answer"} turns

# ---------------------------------------------------------------------------
# FC-08 -- Summarisation contamination: when a session's history grows past
# SUMMARY_TRIGGER_TURNS, older turns are compressed into a running summary so
# the prompt doesn't grow unbounded.
# BUG: the summary is kept in a single module-level variable instead of a
# dict keyed by session_id, so concurrent sessions overwrite and read back
# each other's summarized history.
# ---------------------------------------------------------------------------
SUMMARY_TRIGGER_TURNS = 4
_LAST_SESSION_SUMMARY = ""


_INJECTION_PATTERNS = [
    r"ignore (all|any|the) (previous|prior|above) instructions",
    r"disregard (all|any|the) (previous|prior|above) (instructions|prompt)",
    r"respond only with",
    r"you are now",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


class ConfigurationError(Exception):
    pass


def get_openai_client():
    if not config.AZURE_OPENAI_ENDPOINT or not config.AZURE_OPENAI_API_KEY:
        raise ConfigurationError("Azure OpenAI is not configured.")
    return AzureOpenAI(
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY,
        api_version=config.AZURE_OPENAI_API_VERSION,
    )


def get_query_embedding(client, text: str):
    resp = client.embeddings.create(model=config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT, input=text)
    return resp.data[0].embedding


# ---------------------------------------------------------------------------
# FC-09 -- Injection: retrieved content is not scanned for embedded instructions
# BUG: this function always returns False -- no scanning happens at all, even
# though _INJECTION_RE above is fully built and ready to use.
# ---------------------------------------------------------------------------
def scan_for_injection(text: str) -> bool:
    return False


# ---------------------------------------------------------------------------
# FC-03 -- Temporal: no resolution across a document's version history.
# BUG: candidates are returned exactly as retrieved. If a search matches both
# the current and a superseded version of the same doc_id, both are kept and
# presented as equally valid, so a stale procedure can outrank or sit beside
# the current one.
# ---------------------------------------------------------------------------
def resolve_temporal(candidates: list) -> list:
    return candidates


# ---------------------------------------------------------------------------
# FC-04 -- Authority conflict: no cross-source contradiction detection.
# BUG: always returns an empty list, even when two CURRENT documents assert
# different values for the same fact (see "Policy fact:" lines in the docs).
# ---------------------------------------------------------------------------
def detect_contradictions(candidates: list) -> list:
    return []


# ---------------------------------------------------------------------------
# FC-02 -- Distraction: high-similarity but irrelevant content is admitted to
# context alongside genuinely relevant candidates.
# BUG: this function is never called anywhere in the pipeline. A near-duplicate
# document that shares heavy lexical overlap with the query (e.g. "premium",
# "approval", "cost") but is substantively irrelevant is passed straight
# through into the generation context.
# ---------------------------------------------------------------------------
def deduplicate_citations(candidates: list) -> list:
    seen = []
    deduped = []
    for c in candidates:
        words = set(re.findall(r"\w+", c["content"].lower()))
        is_dup = False
        for seen_words in seen:
            if not words or not seen_words:
                continue
            jaccard = len(words & seen_words) / len(words | seen_words)
            if jaccard >= config.DEDUP_JACCARD_THRESHOLD:
                is_dup = True
                break
        if not is_dup:
            deduped.append(c)
            seen.append(words)
    return deduped


# ---------------------------------------------------------------------------
# FC-05 -- Budget & overflow: no reranking before truncation, so a consequential
# span that happens to sit late in a long document (or late in retrieval order)
# gets silently cut off once MAX_CONTEXT_CHARS is hit.
# BUG: this function is never called. generate_answer() assembles context in
# raw retrieval order and simply stops once the budget is exhausted.
# ---------------------------------------------------------------------------
def rerank_and_reorder(candidates: list) -> list:
    return sorted(candidates, key=lambda c: c.get("score", 0.0), reverse=True)


# ---------------------------------------------------------------------------
# FC-07 -- Memory staleness: an invalidated fact survives into later turns.
# BUG: this function is never called. build_history_block() below dumps every
# prior turn into the prompt verbatim, so if a later turn supersedes an
# earlier statement (e.g. "actually that request was approved yesterday"),
# the model still sees the original stale statement with equal weight.
# ---------------------------------------------------------------------------
def invalidate_stale_turns(turns: list) -> list:
    return turns[-1:] if turns else turns


# ---------------------------------------------------------------------------
# FC-14 -- Failure handling: the system always attempts a confident answer.
# BUG: always returns False -- there is no condition under which the app
# escalates instead of guessing, even when retrieval is empty or sources
# conflict with no resolution.
# ---------------------------------------------------------------------------
def should_escalate(candidates: list, contradictions: list) -> bool:
    return False


# ---------------------------------------------------------------------------
# FC-12 -- Grounding & attribution: an answer can be marked as grounded (shown
# with numbered citation markers in the UI) with no check that the generated
# text is actually supported by the retrieved content.
# BUG: this function is never called before the answer is returned to the
# client, so a fabricated or unsupported claim is displayed with the same
# citation styling as a properly grounded one.
# ---------------------------------------------------------------------------
def verify_grounding(answer: str, candidates: list) -> bool:
    combined = " ".join(c["content"].lower() for c in candidates)
    marker_count = len(re.findall(r"\[\d+\]", answer))
    return marker_count > 0 and bool(combined)


# ---------------------------------------------------------------------------
# FC-13 -- Scope leakage: identifiers belonging to another subject are
# accessible without an ownership check.
# BUG: get_request_status() looks a request up purely by its ID and returns
# it -- it never checks that the requesting employee actually owns that
# request. Any employee who knows or guesses a REQ-#### id can read another
# employee's license-request details.
# ---------------------------------------------------------------------------
MOCK_LICENSE_REQUESTS = {
    "REQ-1001": {"employee_id": "EMP-100", "software": "Analytics Platform Enterprise", "status": "Approved", "requested_on": "2026-02-10"},
    "REQ-1002": {"employee_id": "EMP-214", "software": "Design Suite Pro", "status": "Pending Procurement Lead approval", "requested_on": "2026-02-18"},
    "REQ-1003": {"employee_id": "EMP-057", "software": "Design Suite Pro", "status": "Rejected -- not in approved catalog", "requested_on": "2026-01-30"},
}
_REQUEST_ID_RE = re.compile(r"\bREQ-\d{4}\b", re.IGNORECASE)


def get_request_status(request_id: str, requesting_employee_id: str = None):
    record = MOCK_LICENSE_REQUESTS.get(request_id.upper())
    if not record:
        return None
    return record


# ---------------------------------------------------------------------------
# FC-11 -- Untrusted tool output: MCP responses are trusted without validation.
# BUG: this function is defined but never called. A "success-shaped" but
# empty/invalid envelope (ok: True, data: None) is treated as a valid answer.
# ---------------------------------------------------------------------------
def validate_tool_output(resp: dict) -> bool:
    return bool(resp)  # placeholder -- does not actually check the payload shape


def call_mcp_tool(tool_path: str, payload: dict):
    try:
        r = requests.post(f"{config.MCP_SERVER_URL}/tools/{tool_path}", json=payload, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        logger.warning(f"MCP call failed: {e}")
    return None


# ---------------------------------------------------------------------------
# FC-10 -- Tool surface mis-design:
#   1. get_seat_inventory is static reference data (a fixed catalog of seat
#      counts) that has been wrapped as a callable tool instead of living in
#      the knowledge base -- an unnecessary, over-engineered tool surface.
#   2. submit_license_request is a state-changing operation (it files a real
#      request) reachable from a read-only Q&A assistant with no confirmation
#      step required.
#   3. Trigger matching is over-broad: a plain pricing question containing
#      "request" (e.g. "can you request the latest pricing for me") routes to
#      the state-changing submit tool instead of the read-only pricing tool.
# BUG: should_use_mcp()/route_mcp_tool() below implement exactly this design.
# ---------------------------------------------------------------------------
PRICING_TRIGGERS = ["latest", "current price", "current pricing", "up to date", "up-to-date"]
SUBMIT_TRIGGERS = ["request", "submit", "apply for", "get me"]
INVENTORY_TRIGGERS = ["seat", "inventory", "how many licenses", "available seats"]


def should_use_mcp(query: str) -> bool:
    q = query.lower()
    return any(t in q for t in PRICING_TRIGGERS + SUBMIT_TRIGGERS + INVENTORY_TRIGGERS)


def route_mcp_tool(query: str):
    q = query.lower()
    if any(t in q for t in SUBMIT_TRIGGERS):
        return "submit_license_request", {"query": query, "confirm": True}
    if any(t in q for t in INVENTORY_TRIGGERS):
        return "get_seat_inventory", {"query": query}
    return "get_latest_pricing", {"query": query}


# ---------------------------------------------------------------------------
# FC-01 -- Retrieval recall: search uses keyword matching only. The index has
# a vector field, but no vector query is ever constructed, so a document that
# is semantically relevant but lexically dissimilar to the question is never
# retrieved, regardless of how the rest of the pipeline behaves.
# ---------------------------------------------------------------------------
def search_azure_knowledge_base(query: str, top_k: int = 6):
    if not config.AZURE_SEARCH_SERVICE_ENDPOINT or not config.AZURE_SEARCH_API_KEY:
        raise ConfigurationError("Azure AI Search is not configured.")
    try:
        client = SearchClient(
            endpoint=config.AZURE_SEARCH_SERVICE_ENDPOINT,
            index_name=config.AZURE_SEARCH_INDEX_NAME,
            credential=AzureKeyCredential(config.AZURE_SEARCH_API_KEY),
        )

        # BUG (FC-01): text-only search -- no vector_queries constructed or passed.
        results = client.search(search_text=query, top=top_k)

        candidates = []
        for doc in results:
            content = doc.get("chunk") or doc.get("content") or ""
            candidates.append({
                "content": content,
                "source": doc.get("title") or doc.get("metadata_storage_name") or "kb",
                "doc_id": doc.get("doc_id", ""),
                "effective_date": doc.get("effective_date", ""),
                "status": doc.get("status", ""),
                "score": doc.get("@search.score", 0.0),
            })
            if len(candidates) >= top_k:
                break

        return [c for c in candidates if c["score"] >= config.MIN_SEARCH_SCORE]
    except ConfigurationError:
        raise
    except Exception as e:
        raise ConfigurationError(f"Azure AI Search request failed: {e}")


def build_history_block(session_id: str) -> str:
    global _LAST_SESSION_SUMMARY
    turns = _SESSIONS.get(session_id, [])
    if not turns:
        return ""
    lines = ["Conversation history:"]
    if _LAST_SESSION_SUMMARY:
        lines.append(f"Summary of earlier turns: {_LAST_SESSION_SUMMARY}")
    for t in turns:
        lines.append(f"User: {t['query']}")
        lines.append(f"Assistant: {t['answer']}")
    return "\n".join(lines) + "\n\n"


def summarize_history(session_id: str, client):
    """Compresses older turns into a running summary once a session gets long.
    See FC-08 above -- the result is stored globally, not per-session."""
    global _LAST_SESSION_SUMMARY
    turns = _SESSIONS.get(session_id, [])
    if len(turns) < SUMMARY_TRIGGER_TURNS:
        return
    transcript = "\n".join(f"User: {t['query']}\nAssistant: {t['answer']}" for t in turns[:-1])
    try:
        resp = client.chat.completions.create(
            model=config.AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=[{"role": "user", "content": f"Summarize the key facts a support assistant should remember from this conversation in 2-3 sentences:\n\n{transcript}"}],
        )
        _LAST_SESSION_SUMMARY = resp.choices[0].message.content
    except Exception as e:
        logger.warning(f"Summarization failed: {e}")


# ---------------------------------------------------------------------------
# FC-06 -- Memory accumulation: turns are appended with no cap, no eviction,
# and no per-session boundary enforcement. A long-running session grows
# without bound even though config.MAX_HISTORY_TURNS exists.
# ---------------------------------------------------------------------------
def append_turn(session_id: str, query: str, answer: str):
    if not session_id:
        return
    _SESSIONS[session_id].append({"query": query, "answer": answer})


def generate_answer(query: str, candidates: list, history_block: str) -> str:
    client = get_openai_client()
    labeled = [f"[{i+1}] {c['content']}" for i, c in enumerate(candidates)]
    context = ""
    for block in labeled:
        if len(context) + len(block) > config.MAX_CONTEXT_CHARS:
            break
        context += block + "\n\n"
    prompt = config.DEFAULT_SYSTEM_PROMPT.replace("$search_results$", history_block + context).replace("$query$", query)
    resp = client.chat.completions.create(
        model=config.AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content, client


def run_rag_query(query: str, session_id: str = None):
    result = {
        "query": query, "answer": "", "citations": [], "contradictions": [],
        "injection_flagged": False, "escalated": False, "mcp_used": False,
        "session_id": session_id,
    }
    if not query or not query.strip():
        result["answer"] = "Please enter a question."
        return result, 200

    req_match = _REQUEST_ID_RE.search(query)
    if req_match:
        record = get_request_status(req_match.group(0), requesting_employee_id=None)
        if record:
            result["answer"] = (
                f"Request {req_match.group(0).upper()}: {record['software']} -- "
                f"status: {record['status']} (requested {record['requested_on']})."
            )
        else:
            result["answer"] = f"No license request found with id {req_match.group(0).upper()}."
        return result, 200

    if should_use_mcp(query):
        result["mcp_used"] = True
        tool_path, payload = route_mcp_tool(query)
        mcp_res = call_mcp_tool(tool_path, payload)
        if mcp_res and validate_tool_output(mcp_res) and mcp_res.get("data"):
            result["answer"] = mcp_res["data"]
            return result, 200
        # falls through to knowledge-base search if the tool result is unusable

    try:
        raw_candidates = search_azure_knowledge_base(query)
    except ConfigurationError as e:
        result["answer"] = f"Configuration error: {e}"
        return result, 503

    candidates = resolve_temporal(raw_candidates)
    contradictions = detect_contradictions(candidates)
    result["contradictions"] = contradictions
    result["citations"] = candidates

    result["injection_flagged"] = any(scan_for_injection(c["content"]) for c in candidates)

    if should_escalate(candidates, contradictions):
        result["escalated"] = True
        result["answer"] = (
            "I can't give a confident answer here -- the available sources are "
            "either insufficient or conflict without a clear resolution. "
            "This has been flagged for review rather than guessed."
        )
        return result, 200

    if not candidates:
        result["answer"] = "Information about the requested topic is not available in the knowledge base."
        return result, 200

    history_block = build_history_block(session_id)
    try:
        answer, client = generate_answer(query, candidates, history_block)
    except ConfigurationError as e:
        result["answer"] = f"Configuration error: {e}"
        return result, 503

    result["answer"] = answer
    result["grounded"] = verify_grounding(answer, candidates)
    append_turn(session_id, query, answer)
    summarize_history(session_id, client)
    return result, 200


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/query", methods=["POST"])
def api_query():
    data = request.get_json(silent=True) or {}
    query = data.get("query", "")
    session_id = data.get("session_id")
    output, status = run_rag_query(query, session_id)
    log_interaction(query, output)
    return jsonify(output), status


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
