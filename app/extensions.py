# app/extensions.py
from flask_caching import Cache

# Hier wird das Objekt nur erstellt, aber noch nicht konfiguriert (init_app kommt später)
cache = Cache()
