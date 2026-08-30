from flask import Blueprint

from .api import api_blueprint
from .errors import errors_bp
from .honeypot import honeypot_bp
from .mcp import mcp_bp
from .views import views_bp

# Main Blueprint registered by the Flask application
main_bp = Blueprint("main_aggregator", __name__)

# Register sub-modules
# No URL prefixes are used here to preserve original routes (/webhook, /screener/...)
main_bp.register_blueprint(api_blueprint)
main_bp.register_blueprint(views_bp)
main_bp.register_blueprint(errors_bp)
main_bp.register_blueprint(honeypot_bp)
main_bp.register_blueprint(mcp_bp)
