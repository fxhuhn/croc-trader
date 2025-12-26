import sys
from dataclasses import asdict
from datetime import datetime, timezone
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
    # Allow requests without "application/json" header if the body is valid JSON
    if not request.is_json and not request.data:
        return jsonify({"status": "error", "message": "Missing JSON body"}), 400

    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"status": "error", "message": "Invalid JSON"}), 400

    data["timestamp"] = datetime.now(timezone.utc)
    data["source_ip"] = request.headers.get("X-Real-IP", request.remote_addr)

    try:
        signal = CrocSignal.from_dict(data)
    except (TypeError, KeyError, ValueError) as e:
        return jsonify({"status": "error", "message": f"Invalid data: {str(e)}"}), 400

    # Non-blocking: Put in queue
    current_app.container.queue.put(asdict(signal))

    return jsonify({"status": "queued", "reference": signal.reference}), 202


@bp.post("/toggle-track")
def toggle_track():
    data = request.get_json() or {}
    ...

    result = current_app.container.repo.toggle_trade_tracking(
        symbol=data["symbol"],
        timestamp=data["timestamp"],
        signal=data["signal"],
        timeframe=data.get("timeframe"),
        signal_timestamp=data.get("timestamp"),
    )

    return jsonify(
        {
            "status": "success",
            "tracked": result["tracked"],
            "is_tracked": result["tracked"],
            "trade_id": result["trade_id"],
        }
    )


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
