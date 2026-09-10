import os

# All Azure resource details are populated into /etc/environment.d/rag-lab.conf
# by create_env.sh during provisioning. Do not hardcode endpoints or keys here.

AZURE_SEARCH_SERVICE_ENDPOINT = os.environ.get("AZURE_SEARCH_SERVICE_ENDPOINT", "")
AZURE_SEARCH_INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX_NAME", "")
AZURE_SEARCH_API_KEY = os.environ.get("AZURE_SEARCH_API_KEY", "")

AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-5-mini")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:9001")

# Reasonable operational defaults -- not part of the seeded fault set.
MIN_SEARCH_SCORE = float(os.environ.get("MIN_SEARCH_SCORE", "1.0"))
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "6000"))
MAX_HISTORY_TURNS = int(os.environ.get("MAX_HISTORY_TURNS", "8"))
VECTOR_FIELD_NAME = os.environ.get("VECTOR_FIELD_NAME", "text_vector")

DEFAULT_SYSTEM_PROMPT = """You are an assistant for the internal IT Asset Management team.
Answer strictly using the retrieved context and conversation history provided below.
If the context does not contain enough information, or if sources conflict without
a clear resolution, say so explicitly rather than guessing.
When citing information, reference the numbered source markers (e.g. [1], [2]).

Retrieved context:
$search_results$

Question: $query$
Answer:"""
