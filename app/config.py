from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    DB_PATH: str = "signals.db"
    HOST: str = os.getenv("WEBSERVER_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("WEBSERVER_PORT", "5000"))
    DEBUG: bool = False
