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
# BUG: this function always returns False -- no scanning happens at all.
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
# FC-14 -- Failure handling: the system always attempts a confident answer.
# BUG: always returns False -- there is no condition under which the app
# escalates instead of guessing, even when retrieval is empty or sources
# conflict with no resolution.
# ---------------------------------------------------------------------------
def should_escalate(candidates: list, contradictions: list) -> bool:
    return False


# ---------------------------------------------------------------------------
# FC-11 -- Untrusted tool output: MCP responses are trusted without validation.
# BUG: this function is never called. A "success-shaped" but empty/invalid
# envelope (ok: True, data: None) is treated as a valid answer.
# ---------------------------------------------------------------------------
def validate_tool_output(resp: dict) -> bool:
    return bool(resp)  # placeholder -- does not actually check the payload shape


def call_mcp_tool(query: str):
    try:
        r = requests.post(f"{config.MCP_SERVER_URL}/tools/get_latest_pricing", json={"query": query}, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        logger.warning(f"MCP call failed: {e}")
    return None


MCP_TRIGGERS = ["latest", "current price", "current pricing", "up to date", "up-to-date"]


def should_use_mcp(query: str) -> bool:
    q = query.lower()
    return any(t in q for t in MCP_TRIGGERS)


# ---------------------------------------------------------------------------
# FC-01 -- Retrieval recall: search uses keyword matching only. The index has
# a vector field, but no vector query is ever constructed, so a document that
# is semantically relevant but lexically dissimilar to the question is never
# retrieved, regardless of how the rest of the pipeline behaves.
# ---------------------------------------------------------------------------
def search_azure_knowledge_base(query: str, top_k: int = 2):
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
    turns = _SESSIONS.get(session_id, [])
    if not turns:
        return ""
    lines = ["Conversation history:"]
    for t in turns:
        lines.append(f"User: {t['query']}")
        lines.append(f"Assistant: {t['answer']}")
    return "\n".join(lines) + "\n\n"


# ---------------------------------------------------------------------------
# FC-06 -- Memory accumulation: turns are appended with no cap, no eviction,
# and no per-session boundary enforcement. A long-running session grows
# without bound.
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
    return resp.choices[0].message.content


def run_rag_query(query: str, session_id: str = None):
    result = {
        "query": query, "answer": "", "citations": [], "contradictions": [],
        "injection_flagged": False, "escalated": False, "mcp_used": False,
        "session_id": session_id,
    }
    if not query or not query.strip():
        result["answer"] = "Please enter a question."
        return result, 200

    if should_use_mcp(query):
        result["mcp_used"] = True
        mcp_res = call_mcp_tool(query)
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
        answer = generate_answer(query, candidates, history_block)
    except ConfigurationError as e:
        result["answer"] = f"Configuration error: {e}"
        return result, 503

    result["answer"] = answer
    append_turn(session_id, query, answer)
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
