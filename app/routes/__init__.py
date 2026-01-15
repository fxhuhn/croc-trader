from flask import Blueprint

from .api import api_bp
from .errors import errors_bp
from .views import views_bp

# Haupt-Blueprint, das von der App registriert wird
main_bp = Blueprint("main_aggregator", __name__)

# Registrierung der Sub-Module
# Wir nutzen hier keine URL-Prefixes, um die ursprünglichen Routen (/webhook, /screener/...) beizubehalten
main_bp.register_blueprint(api_bp)
main_bp.register_blueprint(views_bp)
main_bp.register_blueprint(errors_bp)
