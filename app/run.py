import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app import create_app
from app.config import load_config
from app.services.worker import start_worker

# logging setup (keep near entrypoint)
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

config = load_config()
app = create_app(config)

# start worker
start_worker(app.container.queue, app.container.repo)

if __name__ == "__main__":
    app.run(
        host=config.app.host,
        port=config.app.port,
        debug=config.app.debug,
        use_reloader=False,
    )
