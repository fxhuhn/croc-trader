import logging
import logging.config

# forwarded_allow_ips = "*"
# accesslog = "-"


class HealthCheckFilter(logging.Filter):
    def filter(self, record):
        return "GET /health" not in record.getMessage()


# Standardized Format matching app/__init__.py
# Format: HH:MM:SS [LEVEL] name: message
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        }
    },
    "loggers": {
        "gunicorn.error": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "gunicorn.access": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

# Apply immediately so Gunicorn picks it up
logconfig_dict = LOGGING_CONFIG


def on_starting(server):
    """
    Attach filters to loggers.
    """
    # Filter Health Checks from Access Log
    logging.getLogger("gunicorn.access").addFilter(HealthCheckFilter())
