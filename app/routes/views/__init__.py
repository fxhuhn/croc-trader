"""View routes package entry point.

Collects and registers all views and controllers under the views blueprint.
"""

from .blueprint import views_bp

# Import sub-modules to register their respective route endpoints on views_bp
from . import screener, trades, analytics, backtest  # noqa

# Re-expose blueprint for application aggregator registration
__all__ = ["views_bp"]
