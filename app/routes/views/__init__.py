"""View routes package entry point.

Collects and registers all views and controllers under the views blueprint.
"""

# Import sub-modules to register their respective route endpoints on views_bp
from . import analytics, screener, trades  # noqa
from .blueprint import views_bp

# Re-expose blueprint for application aggregator registration
__all__ = ["views_bp"]
