"""
Semantica Explorer : FastAPI Dependencies

Provides ``Depends()``-compatible callables for injecting the
current ``GraphSession`` and ``ConnectionManager`` into route handlers.
"""

from fastapi import Request

from .session import GraphSession
from .ws import ConnectionManager


def get_session(request: Request) -> GraphSession:
    """Retrieve the GraphSession stored on ``app.state``."""
    return request.app.state.session


def get_ws_manager(request: Request) -> ConnectionManager:
    """Retrieve the ConnectionManager stored on ``app.state``."""
    return request.app.state.ws_manager
