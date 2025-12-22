from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    DB_PATH: str = "signals.db"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
