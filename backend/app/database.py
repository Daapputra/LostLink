"""
LostLink Database Module
========================
Setup dan operasi SQLite database untuk menyimpan laporan barang.
"""

import sqlite3
import os
from contextlib import contextmanager
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lostlink.db')


def get_db_path():
    """Return path ke database."""
    return DB_PATH


@contextmanager
def get_db():
    """Context manager untuk koneksi database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Inisialisasi tabel database."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_type TEXT NOT NULL CHECK(report_type IN ('lost', 'found')),
                item_name TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                color TEXT DEFAULT '',
                location TEXT DEFAULT '',
                event_date TEXT DEFAULT '',
                contact TEXT DEFAULT '',
                status TEXT DEFAULT 'active' CHECK(status IN ('active', 'matched', 'resolved', 'closed')),
                created_at TEXT DEFAULT (datetime('now', 'localtime')),
                updated_at TEXT DEFAULT (datetime('now', 'localtime'))
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS match_verifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lost_report_id INTEGER NOT NULL,
                found_report_id INTEGER NOT NULL,
                status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'confirmed', 'rejected')),
                similarity_score REAL DEFAULT 0,
                method TEXT DEFAULT 'tfidf',
                created_at TEXT DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY (lost_report_id) REFERENCES reports(id),
                FOREIGN KEY (found_report_id) REFERENCES reports(id)
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_reports_type ON reports(report_type)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_reports_category ON reports(category)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_reports_status ON reports(status)
        """)

    print(f"[Database] Initialized at {DB_PATH}")


def seed_from_csv(csv_path):
    """Import data dari CSV ke database (jika tabel masih kosong)."""
    import csv

    with get_db() as conn:
        count = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
        if count > 0:
            print(f"[Database] Sudah ada {count} laporan, lewati seeding.")
            return

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in rows:
            conn.execute("""
                INSERT INTO reports (report_type, item_name, category, description, color, location, event_date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                row['report_type'], row['item_name'], row['category'],
                row['description'], row.get('color', ''),
                row.get('location', ''), row.get('event_date', '')
            ))

        print(f"[Database] Seeded {len(rows)} laporan dari {csv_path}")


# --- CRUD Operations ---

def create_report(data: dict) -> int:
    """Buat laporan baru. Return ID laporan."""
    with get_db() as conn:
        cursor = conn.execute("""
            INSERT INTO reports (report_type, item_name, category, description, color, location, event_date, contact)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data['report_type'], data['item_name'], data['category'],
            data['description'], data.get('color', ''),
            data.get('location', ''), data.get('event_date', ''),
            data.get('contact', '')
        ))
        return cursor.lastrowid


def get_report(report_id: int) -> dict:
    """Ambil satu laporan berdasarkan ID."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
        if row:
            return dict(row)
    return None


def get_reports(report_type=None, category=None, status=None,
                search=None, limit=100, offset=0, sort_by='created_at', sort_order='desc'):
    """Ambil daftar laporan dengan filter dan pagination."""
    with get_db() as conn:
        query = "SELECT * FROM reports WHERE 1=1"
        params = []

        if report_type:
            query += " AND report_type = ?"
            params.append(report_type)

        if category:
            query += " AND category = ?"
            params.append(category)

        if status:
            query += " AND status = ?"
            params.append(status)

        if search:
            query += " AND (item_name LIKE ? OR description LIKE ? OR location LIKE ?)"
            search_param = f"%{search}%"
            params.extend([search_param, search_param, search_param])

        # Count total
        count_query = query.replace("SELECT *", "SELECT COUNT(*)")
        total = conn.execute(count_query, params).fetchone()[0]

        # Sorting
        allowed_sort = {'created_at', 'event_date', 'item_name', 'category'}
        if sort_by not in allowed_sort:
            sort_by = 'created_at'
        sort_dir = 'DESC' if sort_order == 'desc' else 'ASC'
        query += f" ORDER BY {sort_by} {sort_dir}"

        # Pagination
        query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = conn.execute(query, params).fetchall()
        return {
            'reports': [dict(r) for r in rows],
            'total': total,
            'limit': limit,
            'offset': offset
        }


def update_report_status(report_id: int, status: str):
    """Update status laporan."""
    with get_db() as conn:
        conn.execute("""
            UPDATE reports SET status = ?, updated_at = datetime('now', 'localtime')
            WHERE id = ?
        """, (status, report_id))


def get_reports_by_type(report_type: str) -> list:
    """Ambil semua laporan berdasarkan tipe (untuk matching)."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM reports WHERE report_type = ? AND status = 'active' ORDER BY created_at DESC",
            (report_type,)
        ).fetchall()
        return [dict(r) for r in rows]


def create_match_verification(lost_id: int, found_id: int, score: float, method: str = 'tfidf'):
    """Buat verifikasi kecocokan."""
    with get_db() as conn:
        conn.execute("""
            INSERT INTO match_verifications (lost_report_id, found_report_id, similarity_score, method)
            VALUES (?, ?, ?, ?)
        """, (lost_id, found_id, score, method))


def get_stats() -> dict:
    """Ambil statistik untuk dashboard."""
    with get_db() as conn:
        total_lost = conn.execute("SELECT COUNT(*) FROM reports WHERE report_type = 'lost'").fetchone()[0]
        total_found = conn.execute("SELECT COUNT(*) FROM reports WHERE report_type = 'found'").fetchone()[0]
        total_matched = conn.execute("SELECT COUNT(*) FROM reports WHERE status = 'matched'").fetchone()[0]
        total_resolved = conn.execute("SELECT COUNT(*) FROM reports WHERE status = 'resolved'").fetchone()[0]

        # Kategori stats
        cat_rows = conn.execute("""
            SELECT category, report_type, COUNT(*) as count
            FROM reports GROUP BY category, report_type
        """).fetchall()

        category_stats = {}
        for row in cat_rows:
            cat = row['category']
            if cat not in category_stats:
                category_stats[cat] = {'lost': 0, 'found': 0}
            category_stats[cat][row['report_type']] = row['count']

        # Aktivitas terbaru
        recent = conn.execute("""
            SELECT id, report_type, item_name, category, location, created_at
            FROM reports ORDER BY created_at DESC LIMIT 10
        """).fetchall()

        # Verifikasi pending
        pending = conn.execute("""
            SELECT COUNT(*) FROM match_verifications WHERE status = 'pending'
        """).fetchone()[0]

        return {
            'total_lost': total_lost,
            'total_found': total_found,
            'total_matched': total_matched,
            'total_resolved': total_resolved,
            'total_reports': total_lost + total_found,
            'pending_verifications': pending,
            'category_stats': category_stats,
            'recent_activity': [dict(r) for r in recent]
        }
