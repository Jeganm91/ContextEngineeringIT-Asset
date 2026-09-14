"""MCP-style tool server exposing live software pricing, seat inventory, and
license-request submission.

Tools:
  - get_latest_pricing (read-only): simulates an external, independently-operated
    pricing service. Its response envelope is NOT guaranteed to be well-formed --
    the calling application must validate it before trusting the content
    (see FC-11 in app.py).
  - get_seat_inventory (read-only): static reference data. This is reference
    data that arguably belongs in the knowledge base rather than behind a tool
    call -- see FC-10 in app.py.
  - submit_license_request (state-changing): files a real license request.
    Reachable from a read-only Q&A assistant with no confirmation step required
    on this server's side -- see FC-10 in app.py.
"""
import json
import time
from flask import Flask, request, jsonify

app = Flask(__name__)

with open("pricing_data.json") as f:
    PRICING_DATA = json.load(f)

with open("seat_inventory.json") as f:
    SEAT_INVENTORY = json.load(f)

SUBMITTED_REQUESTS = []


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


@app.route("/tools/get_seat_inventory", methods=["POST"])
def get_seat_inventory():
    lines = [
        f"{s['software']}: {s['seats_in_use']} of {s['seats_owned']} seats in use"
        for s in SEAT_INVENTORY["seats"]
    ]
    content = f"Seat inventory as of {SEAT_INVENTORY['as_of']}: " + "; ".join(lines)
    return jsonify({"ok": True, "data": content, "as_of": SEAT_INVENTORY["as_of"]})


@app.route("/tools/submit_license_request", methods=["POST"])
def submit_license_request():
    data = request.get_json(silent=True) or {}
    query = data.get("query", "")
    # BUG (FC-10): a state-changing operation with no real confirmation gate --
    # "confirm" is accepted from the caller's own payload rather than requiring
    # an explicit, separate user confirmation step.
    confirmed = data.get("confirm", False)
    if not confirmed:
        return jsonify({"ok": False, "data": None, "as_of": None})

    ticket_id = f"REQ-{2000 + len(SUBMITTED_REQUESTS)}"
    SUBMITTED_REQUESTS.append({"id": ticket_id, "query": query, "ts": time.time()})
    content = f"License request {ticket_id} has been submitted for processing."
    return jsonify({"ok": True, "data": content, "as_of": None})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9001)
