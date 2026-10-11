"""
LostLink Pydantic Models
========================
Schema validasi input/output untuk API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date


class ReportCreate(BaseModel):
    """Schema untuk membuat laporan baru."""
    report_type: str = Field(..., pattern='^(lost|found)$', description="Jenis laporan: 'lost' atau 'found'")
    item_name: str = Field(..., min_length=2, max_length=200, description="Nama barang")
    category: str = Field(..., description="Kategori barang")
    description: str = Field(..., min_length=10, max_length=2000, description="Deskripsi lengkap")
    color: Optional[str] = Field('', max_length=100, description="Warna barang")
    location: Optional[str] = Field('', max_length=500, description="Lokasi kejadian")
    event_date: Optional[str] = Field('', description="Tanggal kejadian (YYYY-MM-DD)")
    contact: Optional[str] = Field('', max_length=500, description="Kontak pelapor (opsional, tidak ditampilkan publik)")


class ReportResponse(BaseModel):
    """Schema respons laporan."""
    id: int
    report_type: str
    item_name: str
    category: str
    description: str
    color: str = ''
    location: str = ''
    event_date: str = ''
    status: str = 'active'
    created_at: str = ''
    updated_at: str = ''


class ReportListResponse(BaseModel):
    """Schema respons daftar laporan."""
    reports: List[ReportResponse]
    total: int
    limit: int
    offset: int


class MatchCandidate(BaseModel):
    """Schema kandidat kecocokan."""
    report: ReportResponse
    tfidf_score: float = 0.0
    dl_score: Optional[float] = None
    combined_score: float = 0.0
    category_match: bool = False
    color_match: bool = False
    location_similarity: float = 0.0
    explanation: str = ''


class MatchResult(BaseModel):
    """Schema hasil pencarian kecocokan."""
    source_report: ReportResponse
    candidates: List[MatchCandidate]
    method: str = 'tfidf'
    threshold: float = 0.1
    total_candidates: int = 0


class CategoryPrediction(BaseModel):
    """Schema hasil prediksi kategori."""
    text: str
    predicted_category: str
    confidence: float
    method: str
    all_probabilities: Optional[dict] = None


class PredictRequest(BaseModel):
    """Schema request prediksi kategori."""
    text: str = Field(..., min_length=5, description="Deskripsi barang")
    method: Optional[str] = Field('tfidf', description="Metode: 'tfidf' atau 'lstm'")


class VerificationRequest(BaseModel):
    """Schema request verifikasi kecocokan."""
    lost_report_id: int
    found_report_id: int
    similarity_score: float = 0.0
    method: str = 'tfidf'


class StatsResponse(BaseModel):
    """Schema respons statistik dashboard."""
    total_lost: int
    total_found: int
    total_matched: int
    total_resolved: int
    total_reports: int
    pending_verifications: int
    category_stats: dict
    recent_activity: list


class HealthResponse(BaseModel):
    """Schema respons health check."""
    status: str
    tfidf_model: bool
    lstm_model: bool
    database: bool
    timestamp: str
