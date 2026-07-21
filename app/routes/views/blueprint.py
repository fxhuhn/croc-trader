"""Flask blueprint definition for the views sub-package."""

from flask import Blueprint, g

views_bp = Blueprint("views", __name__)


@views_bp.before_request
def set_global_last_updated() -> None:
    """Sets the latest signal timestamp on g before each view request."""
    try:
        from .dependencies import _get_trade_view_service

        service = _get_trade_view_service()
        g.last_updated = service.get_latest_signal_date()
    except Exception:
        g.last_updated = None


@views_bp.context_processor
def inject_last_updated() -> dict[str, str | None]:
    """Injects last_updated variable into Jinja template context."""
    return {"last_updated": getattr(g, "last_updated", None)}
