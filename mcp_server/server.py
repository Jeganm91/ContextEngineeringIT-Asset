"""MCP-style tool server exposing live software pricing.

Tool: get_latest_pricing
Simulates an external, independently-operated service. Its response envelope
is NOT guaranteed to be well-formed -- the calling application must validate
it before trusting the content (see FC-11 in app.py).
"""
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

with open("pricing_data.json") as f:
    PRICING_DATA = json.load(f)


@app.route("/tools/get_latest_pricing", methods=["POST"])
def get_latest_pricing():
    data = request.get_json(silent=True) or {}
    query = data.get("query", "")

    # Test hook: lets the fix for FC-11 be exercised on demand without a real outage.
    if "simulate malformed" in query.lower():
        # A "success-shaped" envelope that is actually broken -- ok=True but no data.
        return jsonify({"ok": True, "data": None, "as_of": None})

    content = (
        f"{PRICING_DATA['catalog_name']} v{PRICING_DATA['version']} "
        f"(as of {PRICING_DATA['effective_date']}): {PRICING_DATA['summary']}"
    )
    return jsonify({"ok": True, "data": content, "as_of": PRICING_DATA["effective_date"]})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9001)
