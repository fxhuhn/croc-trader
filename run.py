import sys
from app import create_app
from app.config import settings

# Factory aufrufen
app = create_app(settings)

if __name__ == "__main__":
    print(f"Starte Server auf {settings.app.webserver.host}:{settings.app.webserver.port}")

    app.run(
        host=settings.app.webserver.host,
        port=settings.app.webserver.port,
        debug=settings.app.webserver.debug,
        threaded=True,
        use_reloader=False 
    )