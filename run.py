import logging

from app import create_app
from app.config import settings

# Initialize module-level logger
logger = logging.getLogger(__name__)

app = create_app(settings)


def run_application_server() -> None:
    """Initializes and starts the Flask web server.

    This function encapsulates the application lifecycle to prevent
    unintended side effects and ensure single-threaded initialization
    where required by the EOD engine.
    """
    # application_instance: Flask = create_app(settings)

    logger.info(
        "Starte Server auf %s:%s",
        settings.app.webserver.host,
        settings.app.webserver.port,
    )

    # Note: threaded=True is default in recent Flask/Werkzeug versions suitable for dev
    # application_instance.run(
    app.run(
        host=settings.app.webserver.host,
        port=settings.app.webserver.port,
        debug=settings.app.webserver.debug,
        use_reloader=False,
        threaded=True,
    )


if __name__ == "__main__":
    # Configure basic logging if not already configured in app/__init__ or config
    logging.basicConfig(level=logging.INFO)
    run_application_server()
