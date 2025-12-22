import logging
import os
import queue
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Thread
from typing import Any, List, Optional, Protocol

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / "croc-trader.log"),
    ],
)

logger = logging.getLogger(__name__)


@dataclass
class CrocSignal:
    symbol: str
    signal: str
    timeframe: str
    timestamp: datetime

    close: float
    high: float
    low: float
    wuk: float

    status: str
    kerze: str
    trend: str
    setter: str
    welle: str
    wolke: Optional[str] = None

    strategy_id: Optional[str] = None
    reference: Optional[str] = None

    def __post_init__(self):
        if self.reference is None:
            self.reference = f"{self.symbol}_{self.timestamp.isoformat()}"

    @classmethod
    def from_dict(cls, d: dict) -> "CrocSignal":
        """Create CrocSignal from dictionary, filtering unknown fields"""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered_data)


class QueueProtocol(Protocol):
    def put(self, item: dict[str, Any]) -> None: ...
    def get(self, timeout: float | None = None) -> dict[str, Any]: ...
    def empty(self) -> bool: ...


class SignalDatabase(Protocol):
    def save_signal(self, data: dict[str, Any]) -> None: ...
    def get_signals(
        self,
        symbol: str = None,
        timeframe: str = None,
        signal: str = None,
        limit: int = 100,
    ) -> List[dict]: ...
    def get_distinct(self, column: str) -> List[str]: ...


class LocalQueue:
    def __init__(self):
        self._queue = queue.Queue()

    def put(self, item: dict[str, Any]) -> None:
        self._queue.put(item)

    def get(self, timeout: float | None = None) -> dict[str, Any]:
        return self._queue.get(timeout=timeout)

    def empty(self) -> bool:
        return self._queue.empty()


class SQLiteDatabase:
    def __init__(self, db_path: str = "signals.db"):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        # return rows as dictionaries for easier template usage
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Create tables if they don't exist"""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    close REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    wuk REAL NOT NULL,
                    status TEXT NOT NULL,
                    kerze TEXT NOT NULL,
                    wolke TEXT,
                    trend TEXT NOT NULL,
                    setter TEXT NOT NULL,
                    welle TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_timestamp_covering
                ON signals(symbol, timestamp DESC, signal)
            """)
            conn.commit()
            # --- NEW: Table for Marked Signals ---
            conn.execute("""
                CREATE TABLE IF NOT EXISTS marked_signals (
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    marked_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, timestamp, signal)
                )
            """)
            conn.commit()

            logger.info("Database initialized successfully")

    def save_signal(self, data: dict[str, Any]) -> None:
        """Save signal to database"""
        try:
            logger.info(f"Attempting to save signal: {data}")
            with self._get_conn() as conn:
                conn.execute(
                    """INSERT INTO signals
                    (symbol, timeframe, timestamp, signal, close, high, low, wuk,
                     status, kerze, wolke, trend, setter, welle)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        data["symbol"],
                        data["timeframe"],
                        data["timestamp"].isoformat()
                        if isinstance(data["timestamp"], datetime)
                        else data["timestamp"],
                        data["signal"],
                        data["close"],
                        data["high"],
                        data["low"],
                        data["wuk"],
                        data["status"],
                        data["kerze"],
                        data.get("wolke"),
                        data["trend"],
                        data["setter"],
                        data["welle"],
                    ),
                )
                conn.commit()
            logger.info(f"Signal saved successfully: {data['symbol']}")
        except Exception as e:
            logger.error(f"Database save error: {e}", exc_info=True)
            raise

    def get_signals(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        signal: Optional[str] = None,
        day: Optional[str] = None,
        limit: int = 500,
    ) -> List[dict]:
        """Retrieve filtered signals with marked status"""
        # LEFT JOIN with marked_signals to see which ones are marked
        query = """
            SELECT s.*,
                   CASE WHEN m.symbol IS NOT NULL THEN 1 ELSE 0 END as is_marked
            FROM signals s
            LEFT JOIN marked_signals m
                   ON s.symbol = m.symbol
                  AND s.timestamp = m.timestamp
                  AND s.signal = m.signal
            WHERE 1=1
        """
        params = []

        if symbol:
            query += " AND s.symbol = ?"
            params.append(symbol)
        if timeframe:
            query += " AND s.timeframe = ?"
            params.append(timeframe)
        if signal:
            query += " AND s.signal = ?"
            params.append(signal)
        if day:
            query += " AND date(s.timestamp) = ?"
            params.append(day)

        query += " ORDER BY s.timestamp DESC LIMIT ?"
        params.append(limit)

        try:
            with self._get_conn() as conn:
                cursor = conn.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error fetching signals: {e}")
            return []

    def toggle_mark(self, symbol: str, timestamp: str, signal: str) -> bool:
        """Toggle the marked status of a signal. Returns True if marked, False if unmarked."""
        try:
            with self._get_conn() as conn:
                # Check if exists
                cursor = conn.execute(
                    "SELECT 1 FROM marked_signals WHERE symbol=? AND timestamp=? AND signal=?",
                    (symbol, timestamp, signal),
                )
                exists = cursor.fetchone()

                if exists:
                    conn.execute(
                        "DELETE FROM marked_signals WHERE symbol=? AND timestamp=? AND signal=?",
                        (symbol, timestamp, signal),
                    )
                    conn.commit()
                    return False
                else:
                    conn.execute(
                        "INSERT INTO marked_signals (symbol, timestamp, signal) VALUES (?, ?, ?)",
                        (symbol, timestamp, signal),
                    )
                    conn.commit()
                    return True
        except Exception as e:
            logger.error(f"Error toggling mark: {e}")
            raise

    def get_distinct(self, column: str) -> List[str]:
        """Get distinct values for filters (safeguarded for specific columns)"""
        if column not in ["symbol", "timeframe", "signal"]:
            return []

        try:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    f"SELECT DISTINCT {column} FROM signals ORDER BY {column}"  # nosec B608
                )
                return [row[0] for row in cursor.fetchall() if row[0]]
        except Exception as e:
            logger.error(f"Error fetching distinct {column}: {e}")
            return []


app = Flask(__name__)


class Container:
    def __init__(self, queue: QueueProtocol, repo: SignalDatabase):
        self.queue = queue
        self.repo = repo


# Worker Thread
def process_queue(signal_queue: QueueProtocol, repo: SignalDatabase):
    """Worker thread to process signals from queue"""
    logger.info("Queue worker thread started")
    while True:
        try:
            data = signal_queue.get(timeout=1.0)
            repo.save_signal(data)
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Queue processing error: {e}", exc_info=True)


# Setup container
container = Container(queue=LocalQueue(), repo=SQLiteDatabase("signals.db"))


@app.route("/")
def index():
    return jsonify({"status": "running", "app": "croc-trader", "version": "0.1.1"})


@app.route("/croc-signal")
def dashboard():
    """Dashboard view for signals"""
    # Get filters
    filter_symbol = request.args.get("symbol")
    filter_tf = request.args.get("timeframe")
    filter_sig = request.args.get("signal")
    filter_day = request.args.get("day")  # <--- NEW FILTER

    # Fetch data
    signals = container.repo.get_signals(
        symbol=filter_symbol,
        timeframe=filter_tf,
        signal=filter_sig,
        day=filter_day,
        limit=500,
    )

    # Calculate Stats
    signal_counts = Counter(s["signal"] for s in signals)

    # Fetch dropdown options
    unique_timeframes = container.repo.get_distinct("timeframe")
    unique_signals = container.repo.get_distinct("signal")
    unique_symbols = container.repo.get_distinct("symbol")

    return render_template(
        "croc_signals.html",
        signals=signals,
        stats=signal_counts,
        unique_timeframes=unique_timeframes,
        unique_signals=unique_signals,
        unique_symbols=unique_symbols,
        current_filters={
            "symbol": filter_symbol,
            "timeframe": filter_tf,
            "signal": filter_sig,
            "day": filter_day,  # <--- Pass to template
        },
    )


@app.route("/toggle-mark", methods=["POST"])
def toggle_mark_route():
    data = request.json
    try:
        is_marked = container.repo.toggle_mark(
            data["symbol"], data["timestamp"], data["signal"]
        )
        return jsonify({"status": "success", "is_marked": is_marked})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/webhook", methods=["POST"])
def webhook():
    """Webhook endpoint for TradingView signals"""
    try:
        if not request.is_json:
            return jsonify(
                {"status": "error", "message": "Content-Type must be application/json"}
            ), 400

        data = request.get_json()
        data["timestamp"] = datetime.now(UTC)
        data["source_ip"] = request.headers.get("X-Real-IP", request.remote_addr)

        try:
            signal = CrocSignal.from_dict(data)
            signal_dict = asdict(signal)
            container.queue.put(signal_dict)
            return jsonify({"status": "queued", "reference": signal.reference}), 202

        except TypeError as e:
            logger.error(f"Data validation error: {e}", exc_info=True)
            return jsonify(
                {"status": "error", "message": f"Invalid data: {str(e)}"}
            ), 400

    except Exception as e:
        logger.error(f"Webhook error: {e}", exc_info=True)
        return jsonify({"status": "error", "message": "Internal error"}), 500


# --- NEW: Date Filter ---
@app.template_filter("datetimeformat")
def datetimeformat_filter(value, format="%Y-%m-%d %H:%M"):
    """Format date string or object to requested format"""
    if not value:
        return ""
    try:
        # If it's a string, try to parse ISO format
        if isinstance(value, str):
            # Handle potential Z suffix or simple iso format
            value = value.replace("Z", "+00:00")
            dt = datetime.fromisoformat(value)
        else:
            dt = value
        return dt.strftime(format)
    except Exception:
        return value


@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200


def start_worker():
    """Start queue worker thread"""
    worker = Thread(
        target=process_queue,
        args=(container.queue, container.repo),
        daemon=True,
        name="QueueWorker",
    )
    worker.start()
    logger.info("Worker thread started")


if __name__ == "__main__":
    load_dotenv()
    logger.info("=" * 80)
    logger.info("croc-trader - starting up")
    logger.info("=" * 80)

    start_worker()
    app.run(
        host=os.getenv("WEBSERVER_HOST", "127.0.0.1"),
        port=os.getenv("WEBSERVER_PORT", 5000),
        debug=os.getenv("WEBSERVER_DEBUG", False),
        use_reloader=False,
    )
