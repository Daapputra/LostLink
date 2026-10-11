import React, { useState, useEffect } from 'react';
import { 
  Search, 
  Filter, 
  MapPin, 
  Calendar, 
  Sparkles, 
  HelpCircle, 
  CheckCircle,
  Tag
} from 'lucide-react';
import { api } from '../services/api';

export default function ReportsPage({ onMatchItem }) {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [reportType, setReportType] = useState('');
  const [category, setCategory] = useState('');
  const [search, setSearch] = useState('');

  const categories = [
    'Semua Kategori',
    'Dompet',
    'Tas',
    'Elektronik',
    'Dokumen & Kartu',
    'Kunci',
    'Aksesori',
    'Alat Tulis',
    'Laptop',
    'Handphone',
    'Barang Lainnya'
  ];

  const fetchReports = async () => {
    setLoading(true);
    const data = await api.getReports({
      report_type: reportType,
      category: category === 'Semua Kategori' ? '' : category,
      search: search
    });
    setReports(data.reports || []);
    setLoading(false);
  };

  useEffect(() => {
    fetchReports();
  }, [reportType, category]);

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    fetchReports();
  };

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Daftar Laporan Kampus</h1>
          <p className="page-desc">Eksplorasi laporan barang hilang dan temuan di seluruh area kampus</p>
        </div>
      </div>

      {/* Filters & Search Bar */}
      <div className="glass-card" style={{ marginBottom: 24, padding: 18 }}>
        <form onSubmit={handleSearchSubmit} style={{ display: 'flex', flexWrap: 'wrap', gap: 14, alignItems: 'center' }}>
          
          {/* Type Filter Buttons */}
          <div style={{ display: 'flex', gap: 6, background: 'rgba(15,23,42,0.8)', padding: 4, borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)' }}>
            <button
              type="button"
              onClick={() => setReportType('')}
              style={{
                padding: '8px 14px',
                borderRadius: 'var(--radius-sm)',
                border: 'none',
                background: reportType === '' ? 'var(--primary)' : 'transparent',
                color: '#fff',
                fontSize: 13,
                fontWeight: 600,
                cursor: 'pointer'
              }}
            >
              Semua
            </button>
            <button
              type="button"
              onClick={() => setReportType('lost')}
              style={{
                padding: '8px 14px',
                borderRadius: 'var(--radius-sm)',
                border: 'none',
                background: reportType === 'lost' ? '#F43F5E' : 'transparent',
                color: '#fff',
                fontSize: 13,
                fontWeight: 600,
                cursor: 'pointer'
              }}
            >
              Barang Hilang
            </button>
            <button
              type="button"
              onClick={() => setReportType('found')}
              style={{
                padding: '8px 14px',
                borderRadius: 'var(--radius-sm)',
                border: 'none',
                background: reportType === 'found' ? '#10B981' : 'transparent',
                color: '#fff',
                fontSize: 13,
                fontWeight: 600,
                cursor: 'pointer'
              }}
            >
              Barang Temuan
            </button>
          </div>

          {/* Category Dropdown */}
          <select 
            value={category} 
            onChange={(e) => setCategory(e.target.value)}
            className="form-select"
            style={{ width: 'auto', minWidth: 170 }}
          >
            {categories.map((cat, idx) => (
              <option key={idx} value={cat}>{cat}</option>
            ))}
          </select>

          {/* Search Input */}
          <div style={{ flex: 1, minWidth: 220, position: 'relative' }}>
            <input 
              type="text"
              placeholder="Cari kata kunci (e.g. dompet biru, perpus, Asus)..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="form-input"
              style={{ paddingLeft: 40 }}
            />
            <Search size={18} style={{ position: 'absolute', left: 14, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-dim)' }} />
          </div>

          <button type="submit" className="btn btn-secondary">
            Cari
          </button>
        </form>
      </div>

      {/* Reports Grid */}
      {loading ? (
        <div style={{ textAlign: 'center', padding: '60px 0', color: 'var(--text-muted)' }}>
          <div style={{ fontSize: 32, marginBottom: 12 }}>⏳</div>
          <div>Memuat data laporan...</div>
        </div>
      ) : reports.length === 0 ? (
        <div className="glass-card" style={{ textAlign: 'center', padding: '60px 20px' }}>
          <p style={{ color: 'var(--text-muted)', fontSize: 16 }}>Tidak ada laporan yang sesuai kriteria pencarian.</p>
        </div>
      ) : (
        <div className="reports-grid">
          {reports.map((r) => (
            <div key={r.id} className="report-card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                <span className={`badge ${r.report_type === 'lost' ? 'badge-lost' : 'badge-found'}`}>
                  {r.report_type === 'lost' ? <HelpCircle size={12} /> : <CheckCircle size={12} />}
                  {r.report_type === 'lost' ? 'HILANG' : 'DITEMUKAN'}
                </span>
                <span style={{ fontSize: 12, color: 'var(--text-dim)' }}>
                  ID #{r.id}
                </span>
              </div>

              <h3 style={{ fontSize: 17, fontWeight: 700, color: '#fff', marginBottom: 6 }}>
                {r.item_name}
              </h3>

              <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: '#818CF8', marginBottom: 10, fontWeight: 600 }}>
                <Tag size={13} /> {r.category} {r.color ? `• Warna: ${r.color}` : ''}
              </div>

              <p style={{ fontSize: 13, color: 'var(--text-muted)', flex: 1, marginBottom: 16, lineHeight: 1.5 }}>
                {r.description}
              </p>

              <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: 12, fontSize: 12, color: 'var(--text-dim)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                  <MapPin size={13} /> {r.location || 'Area Kampus'}
                </span>
                <button 
                  className="btn btn-secondary" 
                  style={{ padding: '6px 12px', fontSize: 12, color: '#818CF8' }}
                  onClick={() => onMatchItem(r)}
                >
                  <Sparkles size={13} /> Cari Cocok
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
