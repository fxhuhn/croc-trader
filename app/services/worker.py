from __future__ import annotations

import logging
import queue
import sys
from pathlib import Path
from threading import Thread

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from app.core.protocols import QueueProtocol, SignalRepository

logger = logging.getLogger(__name__)


def process_queue(signal_queue: QueueProtocol, repo: SignalRepository) -> None:
    logger.info("Queue worker thread started")
    while True:
        try:
            data = signal_queue.get(timeout=1.0)
            repo.save_signal(data)
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Queue processing error: {e}", exc_info=True)


def start_worker(signal_queue: QueueProtocol, repo: SignalRepository) -> Thread:
    worker = Thread(
        target=process_queue,
        args=(signal_queue, repo),
        daemon=True,
        name="QueueWorker",
    )
    worker.start()
    logger.info("Worker thread started")
    return worker
