import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from app.core.dataclasses import CrocSignal

bp = Blueprint("api", __name__)


@bp.get("/health")
def health():
    return jsonify({"status": "healthy"}), 200


@bp.post("/webhook")
def webhook():
    if not request.is_json:
        return jsonify(
            {"status": "error", "message": "Content-Type must be application/json"}
        ), 400

    data = request.get_json()
    data["timestamp"] = datetime.now(UTC)
    data["source_ip"] = request.headers.get("X-Real-IP", request.remote_addr)

    try:
        signal = CrocSignal.from_dict(data)
    except TypeError as e:
        return jsonify({"status": "error", "message": f"Invalid data: {str(e)}"}), 400

    current_app.container.queue.put(asdict(signal))
    return jsonify({"status": "queued", "reference": signal.reference}), 202


@bp.post("/toggle-track")
def toggle_track():
    payload = request.get_json(force=True)
    res = current_app.container.repo.toggle_trade_tracking(
        payload["symbol"], payload["timestamp"], payload["signal"]
    )
    return jsonify({"status": "success", **res})


@bp.get("/get-trade/<int:trade_id>")
def get_trade(trade_id: int):
    trade = current_app.container.repo.get_trade(trade_id)
    if not trade:
        return jsonify({"status": "error", "message": "Trade not found"}), 404
    return jsonify({"status": "success", "trade": trade})


@bp.post("/update-trade/<int:trade_id>")
def update_trade(trade_id: int):
    payload = request.get_json(force=True)
    ok = current_app.container.repo.update_trade(trade_id, payload)
    if not ok:
        return jsonify({"status": "error", "message": "No fields to update"}), 400
    return jsonify({"status": "success"})
