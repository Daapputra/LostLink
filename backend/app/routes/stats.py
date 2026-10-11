"""
LostLink Stats Router
======================
Endpoints untuk statistik ringkasan dashboard sistem.
"""

from fastapi import APIRouter
from ..models import StatsResponse
from ..database import get_stats

router = APIRouter(prefix="/stats", tags=["Statistics"])


@router.get("", response_model=StatsResponse)
def get_dashboard_stats():
    """Ambil statistik ringkasan untuk dashboard LostLink."""
    stats = get_stats()
    return stats
