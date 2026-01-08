# gunicorn_conf.py
import logging

# accesslog = "-"
# access_log_format = (
#    '%(t)s [gunicorn] %({x-forwarded-for}i)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'
# )


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
