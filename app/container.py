import queue
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.core.protocols import QueueProtocol, SignalRepository
from app.database.sqlite_repo import SQLiteRepository


class LocalQueue(QueueProtocol):
    def __init__(self):
        self._q: queue.Queue[dict[str, Any]] = queue.Queue()

    def put(self, item: dict[str, Any]) -> None:
        self._q.put(item)

    def get(self, timeout: float | None = None) -> dict[str, Any]:
        return self._q.get(timeout=timeout)

    def empty(self) -> bool:
        return self._q.empty()


@dataclass
class Container:
    queue: QueueProtocol
    repo: SignalRepository


def build_container(db_path: str = "signals.db") -> Container:
    return Container(queue=LocalQueue(), repo=SQLiteRepository(db_path))
