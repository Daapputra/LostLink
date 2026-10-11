"""
LostLink Matches Router
========================
Endpoints untuk AI matching pencocokan barang hilang dan barang temuan.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from ..models import MatchResult, MatchCandidate, VerificationRequest, ReportResponse
from ..database import (
    get_report, get_reports_by_type, create_match_verification, update_report_status
)

router = APIRouter(prefix="/matches", tags=["Matching"])

# Model holders will be injected / accessed via singleton or main app state
tfidf_instance = None
dl_instance = None


def set_models(tfidf_model, dl_model):
    """Set model NLP instances."""
    global tfidf_instance, dl_instance
    tfidf_instance = tfidf_model
    dl_instance = dl_model


@router.get("/find/{report_id}", response_model=MatchResult)
def find_matches_for_report(
    report_id: int,
    method: str = Query("tfidf", pattern="^(tfidf|dl|combined)$"),
    threshold: float = Query(0.05, ge=0.0, le=1.0),
    top_k: int = Query(10, ge=1, le=50)
):
    """
    Cari laporan yang berpotensi cocok untuk suatu laporan tertentu.
    Jika laporan adalah 'lost', cari di 'found'. Jika 'found', cari di 'lost'.
    """
    source_report = get_report(report_id)
    if not source_report:
        raise HTTPException(status_code=404, detail="Laporan sumber tidak ditemukan")

    # Tentukan tipe lawan
    target_type = "found" if source_report["report_type"] == "lost" else "lost"
    candidates = get_reports_by_type(target_type)

    if not candidates:
        return {
            "source_report": source_report,
            "candidates": [],
            "method": method,
            "threshold": threshold,
            "total_candidates": 0
        }

    matched_list = []
    if tfidf_instance and tfidf_instance.is_loaded:
        raw_matches = tfidf_instance.find_matches(
            source_report=source_report,
            candidate_reports=candidates,
            threshold=threshold,
            top_k=top_k
        )
        for m in raw_matches:
            matched_list.append(MatchCandidate(
                report=ReportResponse(**m["report"]),
                tfidf_score=m.get("tfidf_score", 0.0),
                dl_score=m.get("dl_score"),
                combined_score=m.get("combined_score", 0.0),
                category_match=m.get("category_match", False),
                color_match=m.get("color_match", False),
                location_similarity=m.get("location_similarity", 0.0),
                explanation=m.get("explanation", "")
            ))
    else:
        # Fallback jika model belum di-train: simple word overlap
        for c in candidates:
            source_words = set(source_report.get("description", "").lower().split())
            cand_words = set(c.get("description", "").lower().split())
            overlap = len(source_words & cand_words)
            sim = overlap / max(len(source_words | cand_words), 1)
            cat_match = source_report.get("category") == c.get("category")
            if sim >= threshold or cat_match:
                combined = sim * 0.7 + (0.3 if cat_match else 0.0)
                matched_list.append(MatchCandidate(
                    report=ReportResponse(**c),
                    tfidf_score=round(sim, 4),
                    combined_score=round(combined, 4),
                    category_match=cat_match,
                    color_match=source_report.get("color", "") == c.get("color", "") and bool(source_report.get("color")),
                    location_similarity=0.0,
                    explanation=f"Overlap kata: {overlap} kata ({round(sim*100, 1)}%)"
                ))
        matched_list.sort(key=lambda x: x.combined_score, reverse=True)
        matched_list = matched_list[:top_k]

    return MatchResult(
        source_report=ReportResponse(**source_report),
        candidates=matched_list,
        method=method,
        threshold=threshold,
        total_candidates=len(matched_list)
    )


@router.post("/verify")
def verify_match(data: VerificationRequest):
    """
    Verifikasi kecocokan antara laporan barang hilang dan laporan barang temuan.
    Jika dikonfirmasi, status kedua laporan diupdate ke 'matched'.
    """
    lost_rep = get_report(data.lost_report_id)
    found_rep = get_report(data.found_report_id)

    if not lost_rep or not found_rep:
        raise HTTPException(status_code=404, detail="Laporan tidak ditemukan")

    create_match_verification(
        lost_id=data.lost_report_id,
        found_id=data.found_report_id,
        score=data.similarity_score,
        method=data.method
    )

    update_report_status(data.lost_report_id, "matched")
    update_report_status(data.found_report_id, "matched")

    return {
        "status": "success",
        "message": f"Kecocokan berhasil diverifikasi untuk laporan #{data.lost_report_id} dan #{data.found_report_id}",
        "lost_report_id": data.lost_report_id,
        "found_report_id": data.found_report_id
    }
