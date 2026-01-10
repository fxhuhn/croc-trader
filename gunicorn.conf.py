import logging

forwarded_allow_ips = "*"


class HealthCheckFilter(logging.Filter):
    def filter(self, record):
        # Filtert Zeilen raus, die "GET /health" enthalten
        return "GET /health" not in record.getMessage()


def on_starting(server):
    """
    Diese Funktion wird von Gunicorn automatisch beim Start ausgeführt.
    Wir hängen unseren Filter an den 'gunicorn.access' Logger.
    """
    access_logger = logging.getLogger("gunicorn.access")
    access_logger.addFilter(HealthCheckFilter())
