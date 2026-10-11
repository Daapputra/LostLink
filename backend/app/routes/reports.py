"""
LostLink Reports Router
========================
Endpoints untuk CRUD laporan barang hilang dan barang temuan.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from ..models import ReportCreate, ReportResponse, ReportListResponse
from ..database import (
    create_report, get_report, get_reports, update_report_status
)

router = APIRouter(prefix="/reports", tags=["Reports"])


@router.get("", response_model=ReportListResponse)
def list_reports(
    report_type: Optional[str] = Query(None, pattern="^(lost|found)$"),
    category: Optional[str] = None,
    status: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("created_at", pattern="^(created_at|event_date|item_name|category)$"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$")
):
    """Ambil daftar laporan dengan filtering, searching, sorting, dan pagination."""
    result = get_reports(
        report_type=report_type,
        category=category,
        status=status,
        search=search,
        limit=limit,
        offset=offset,
        sort_by=sort_by,
        sort_order=sort_order
    )
    return result


@router.post("", response_model=ReportResponse, status_code=201)
def add_report(report_data: ReportCreate):
    """Buat laporan baru (barang hilang atau barang temuan)."""
    report_dict = report_data.dict()
    report_id = create_report(report_dict)
    created = get_report(report_id)
    if not created:
        raise HTTPException(status_code=500, detail="Gagal membuat laporan")
    return created


@router.get("/{report_id}", response_model=ReportResponse)
def get_report_detail(report_id: int):
    """Ambil detail satu laporan berdasarkan ID."""
    report = get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Laporan tidak ditemukan")
    return report


@router.patch("/{report_id}/status")
def change_report_status(report_id: int, status: str = Query(..., pattern="^(active|matched|resolved|closed)$")):
    """Perbarui status laporan (active, matched, resolved, closed)."""
    report = get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Laporan tidak ditemukan")
    update_report_status(report_id, status)
    return {"message": "Status berhasil diperbarui", "id": report_id, "new_status": status}
