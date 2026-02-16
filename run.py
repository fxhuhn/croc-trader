import logging
from app import create_app
from app.config import settings

# Initialize module-level logger
logger = logging.getLogger(__name__)

# Instantiate the application for WSGI servers (Gunicorn)
app = create_app(settings)

def run_application_server() -> None:
    """Manual entry point for starting the Flask development server."""
    logger.info(
        "Starte Server auf %s:%s", 
        settings.app.webserver.host, 
        settings.app.webserver.port
    )

    app.run(
        host=settings.app.webserver.host,
        port=settings.app.webserver.port,
        debug=settings.app.webserver.debug,
        use_reloader=False, 
        threaded=True 
    )

if __name__ == "__main__":
    # Configure basic logging if not already configured in app/__init__ or config
    logging.basicConfig(level=logging.INFO)
    run_application_server()