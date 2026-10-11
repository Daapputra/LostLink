"""
LostLink Backend Application Entry Point
==========================================
FastAPI application untuk LostLink: AI-Powered Campus Lost & Found Agent.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from .database import init_db, seed_from_csv
from .nlp.tfidf_model import TFIDFModel
from .nlp.dl_model import DLModel
from .routes import reports, matches, predict, stats
from .models import HealthResponse

# Global model instances
tfidf_model = TFIDFModel()
dl_model = DLModel()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle startup and shutdown handler."""
    print("=" * 60)
    print("Starting LostLink Backend Server...")
    print("=" * 60)

    # 1. Initialize SQLite Database
    init_db()

    # 2. Seed database from CSV if needed
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(os.path.dirname(base_dir), 'dataset', 'lostlink_dataset.csv')
    if os.path.exists(csv_path):
        seed_from_csv(csv_path)

    # 3. Load NLP Models
    tfidf_loaded = tfidf_model.load()
    dl_loaded = dl_model.load()

    # Pass models to route modules
    matches.set_models(tfidf_model, dl_model)
    predict.set_models(tfidf_model, dl_model)

    print(f"[Model Status] TF-IDF Loaded: {tfidf_loaded} | DL/LSTM Loaded: {dl_loaded}")
    print("LostLink API is ready!")
    yield
    print("Shutting down LostLink Backend Server...")


app = FastAPI(
    title="LostLink API — AI-Powered Campus Lost & Found Agent",
    description="Sistem pencarian dan pencocokan barang hilang dan barang temuan di lingkungan kampus menggunakan NLP (TF-IDF vs LSTM).",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(reports.router, prefix="/api")
app.include_router(matches.router, prefix="/api")
app.include_router(predict.router, prefix="/api")
app.include_router(stats.router, prefix="/api")


@app.get("/")
def root():
    return {
        "name": "LostLink API",
        "description": "AI-Powered Campus Lost & Found Agent",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "reports": "/api/reports",
            "matches": "/api/matches/find/{report_id}",
            "predict_category": "/api/predict/category",
            "nlp_comparison": "/api/predict/model-info",
            "stats": "/api/stats",
            "health": "/health"
        }
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint untuk memverifikasi kesiapan sistem dan model."""
    return HealthResponse(
        status="healthy",
        tfidf_model=tfidf_model.is_loaded,
        lstm_model=dl_model.is_loaded,
        database=True,
        timestamp=datetime.now().isoformat()
    )
