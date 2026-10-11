import React, { useState, useEffect } from 'react';
import { 
  Sparkles, 
  Search, 
  HelpCircle, 
  CheckCircle, 
  ArrowRight, 
  Check, 
  MapPin, 
  Tag, 
  Percent, 
  AlertCircle
} from 'lucide-react';
import { api } from '../services/api';

export default function MatchingPage({ initialItem }) {
  const [lostReports, setLostReports] = useState([]);
  const [selectedReport, setSelectedReport] = useState(initialItem || null);
  const [matches, setMatches] = useState([]);
  const [loading, setLoading] = useState(false);
  const [method, setMethod] = useState('tfidf');
  const [verifiedList, setVerifiedList] = useState([]);

  useEffect(() => {
    async function loadReports() {
      const data = await api.getReports({ report_type: 'lost', limit: 50 });
      setLostReports(data.reports || []);
      if (!selectedReport && data.reports && data.reports.length > 0) {
        setSelectedReport(data.reports[0]);
      }
    }
    loadReports();
  }, []);

  useEffect(() => {
    if (selectedReport) {
      runMatching(selectedReport.id);
    }
  }, [selectedReport, method]);

  const runMatching = async (reportId) => {
    setLoading(true);
    try {
      const res = await api.findMatches(reportId, method);
      setMatches(res.candidates || []);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const handleVerify = async (candidate) => {
    if (!selectedReport) return;
    await api.verifyMatch(selectedReport.id, candidate.report.id, candidate.combined_score);
    setVerifiedList(prev => [...prev, candidate.report.id]);
  };

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">AI Matching Center (Agentic Reconciler)</h1>
          <p className="page-desc">Mencocokkan deskripsi barang hilang dengan data barang temuan menggunakan TF-IDF Cosine Similarity & Multi-Feature Weighting</p>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '340px 1fr', gap: 24 }}>
        
        {/* Left: Source Item Selector */}
        <div>
          <div className="glass-card" style={{ padding: 18, marginBottom: 16 }}>
            <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
              <HelpCircle size={18} color="#F43F5E" /> Pilih Barang Hilang
            </h3>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 420, overflowY: 'auto' }}>
              {lostReports.map((item) => {
                const isSelected = selectedReport?.id === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => setSelectedReport(item)}
                    style={{
                      padding: '12px 14px',
                      borderRadius: 'var(--radius-md)',
                      border: isSelected ? '1px solid var(--primary)' : '1px solid var(--border-color)',
                      background: isSelected ? 'rgba(99,102,241,0.15)' : 'rgba(255,255,255,0.02)',
                      color: isSelected ? '#fff' : 'var(--text-muted)',
                      textAlign: 'left',
                      cursor: 'pointer',
                      transition: 'all 0.2s'
                    }}
                  >
                    <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 2 }}>{item.item_name}</div>
                    <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>
                      #{item.id} • {item.category} • {item.location || 'Kampus'}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Method Selector */}
          <div className="glass-card" style={{ padding: 18 }}>
            <label className="form-label">Metode Pencocokan NLP</label>
            <select
              value={method}
              onChange={(e) => setMethod(e.target.value)}
              className="form-select"
            >
              <option value="tfidf">TF-IDF + Cosine Similarity (Default)</option>
              <option value="combined">Multi-Feature Hybrid Weighting</option>
            </select>
            <p style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 8 }}>
              Bobot scoring: Deskripsi ($50\%$) + Kategori ($20\%$) + Warna ($15\%$) + Lokasi ($15\%$).
            </p>
          </div>
        </div>

        {/* Right: Matching Results */}
        <div>
          {selectedReport && (
            <div className="glass-card" style={{ marginBottom: 20, background: 'rgba(99,102,241,0.08)', borderColor: 'rgba(99,102,241,0.3)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                  <span className="badge badge-lost" style={{ marginBottom: 8 }}>Barang yang Dicari (Lost Item)</span>
                  <h2 style={{ fontSize: 20, fontWeight: 700, color: '#fff' }}>{selectedReport.item_name}</h2>
                  <p style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 4 }}>
                    "{selectedReport.description}"
                  </p>
                  <div style={{ display: 'flex', gap: 14, fontSize: 12, color: '#A5B4FC', marginTop: 10 }}>
                    <span>🏷️ Kategori: {selectedReport.category}</span>
                    <span>🎨 Warna: {selectedReport.color || '-'}</span>
                    <span>📍 Lokasi: {selectedReport.location || '-'}</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          <div style={{ marginBottom: 14, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3 style={{ fontSize: 16, fontWeight: 700, color: '#fff' }}>
              Rekomendasi Kandidat Barang Temuan ({matches.length} Ditemukan)
            </h3>
            {loading && <span style={{ fontSize: 12, color: '#818CF8' }}>⚡ Sedang menghitung similarity...</span>}
          </div>

          {loading ? (
            <div style={{ textAlign: 'center', padding: '50px 0', color: 'var(--text-muted)' }}>
              Menjalankan NLP similarity matching...
            </div>
          ) : matches.length === 0 ? (
            <div className="glass-card" style={{ textAlign: 'center', padding: '40px 20px' }}>
              <AlertCircle size={32} color="#F59E0B" style={{ margin: '0 auto 10px auto' }} />
              <h4 style={{ fontSize: 16, fontWeight: 600 }}>Belum Ada Laporan yang Cocok</h4>
              <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>
                Tidak ditemukan laporan barang temuan dengan skor kemiripan di atas ambang batas.
              </p>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              {matches.map((cand, idx) => {
                const rep = cand.report;
                const isVerified = verifiedList.includes(rep.id);
                const scorePct = Math.round(cand.combined_score * 100);

                return (
                  <div key={idx} className="glass-card" style={{ borderLeft: scorePct > 70 ? '4px solid #10B981' : '4px solid #6366F1' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
                      <div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                          <span className="badge badge-found">BARANG TEMUAN #{rep.id}</span>
                          <span style={{ fontWeight: 700, fontSize: 16, color: '#fff' }}>{rep.item_name}</span>
                        </div>
                        <div style={{ fontSize: 12, color: 'var(--text-dim)', display: 'flex', gap: 12 }}>
                          <span>📍 {rep.location || 'Kampus'}</span>
                          <span>🏷️ {rep.category}</span>
                          <span>🎨 {rep.color || '-'}</span>
                        </div>
                      </div>

                      <div style={{ textAlign: 'right' }}>
                        <div className="match-score-badge">
                          <Percent size={16} /> {scorePct}% Cocok
                        </div>
                        <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 4 }}>
                          TF-IDF: {Math.round(cand.tfidf_score * 100)}%
                        </div>
                      </div>
                    </div>

                    <p style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 14, background: 'rgba(0,0,0,0.2)', padding: 12, borderRadius: 8 }}>
                      "{rep.description}"
                    </p>

                    {/* Breakdown details */}
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 14, fontSize: 12 }}>
                      <span style={{ padding: '3px 8px', borderRadius: 6, background: cand.category_match ? 'rgba(16,185,129,0.15)' : 'rgba(255,255,255,0.05)', color: cand.category_match ? '#34D399' : 'var(--text-dim)' }}>
                        {cand.category_match ? '✓ Kategori Sama' : '✗ Kategori Beda'}
                      </span>
                      <span style={{ padding: '3px 8px', borderRadius: 6, background: cand.color_match ? 'rgba(16,185,129,0.15)' : 'rgba(255,255,255,0.05)', color: cand.color_match ? '#34D399' : 'var(--text-dim)' }}>
                        {cand.color_match ? '✓ Warna Cocok' : '✗ Warna Beda'}
                      </span>
                      {cand.location_similarity > 0 && (
                        <span style={{ padding: '3px 8px', borderRadius: 6, background: 'rgba(6,182,212,0.15)', color: '#22D3EE' }}>
                          📍 Kesamaan Lokasi: {Math.round(cand.location_similarity * 100)}%
                        </span>
                      )}
                    </div>

                    {cand.explanation && (
                      <div style={{ fontSize: 12, color: '#A5B4FC', marginBottom: 14, fontStyle: 'italic' }}>
                        💡 Alasan AI: {cand.explanation}
                      </div>
                    )}

                    <div style={{ display: 'flex', justifyContent: 'flex-end', borderTop: '1px solid var(--border-color)', paddingTop: 12 }}>
                      {isVerified ? (
                        <span className="btn btn-secondary" style={{ color: '#10B981', cursor: 'default' }}>
                          <Check size={16} /> Telah Dikonfirmasi Cocok
                        </span>
                      ) : (
                        <button className="btn btn-success" onClick={() => handleVerify(cand)}>
                          <Check size={16} /> Verifikasi Kecocokan Ini
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

      </div>
    </div>
  );
}
