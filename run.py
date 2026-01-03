import signal
import sys

from app import create_app
from app.config import settings

# Factory aufrufen
app = create_app(settings)


def graceful_shutdown(signum, frame):
    print("Signal empfangen. Beende Hintergrund-Dienste sauber...")

    # 1. Zugriff auf Worker holen
    worker = app.extensions.get("webhook_worker")
    if worker:
        # Stop-Signal an Thread senden
        worker._stop_event.set()

        # WICHTIG: Queue leeren!
        # Wir warten kurz, damit der Thread den aktuellen Batch schreiben kann
        print("Warte auf Queue-Verarbeitung...")
        worker._thread.join(timeout=5.0)

    print("Shutdown complete.")
    sys.exit(0)


if __name__ == "__main__":
    # Signale registrieren (für Docker stop)
    signal.signal(signal.SIGTERM, graceful_shutdown)
    signal.signal(signal.SIGINT, graceful_shutdown)

    print(
        f"Starte Server auf {settings.app.webserver.host}:{settings.app.webserver.port}"
    )

    app.run(
        host=settings.app.webserver.host,
        port=settings.app.webserver.port,
        debug=settings.app.webserver.debug,
        threaded=True,
        use_reloader=False,  # Wichtig: False, damit der Worker-Thread nicht 2x startet im Debug-Modus
    )
