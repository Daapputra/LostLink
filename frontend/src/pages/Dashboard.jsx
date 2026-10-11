import React from 'react';
import { 
  Search, 
  CheckCircle, 
  HelpCircle, 
  Sparkles, 
  ArrowRight, 
  Clock, 
  MapPin, 
  Tag,
  Cpu
} from 'lucide-react';

export default function Dashboard({ stats, onNavigate }) {
  const statItems = [
    { label: 'Barang Hilang (Lost)', value: stats?.total_lost || 0, icon: HelpCircle, color: '#F43F5E', bg: 'rgba(244,63,94,0.15)' },
    { label: 'Barang Ditemukan (Found)', value: stats?.total_found || 0, icon: CheckCircle, color: '#10B981', bg: 'rgba(16,185,129,0.15)' },
    { label: 'Kecocokan AI (Matched)', value: stats?.total_matched || 0, icon: Sparkles, color: '#F59E0B', bg: 'rgba(245,158,11,0.15)' },
    { label: 'Kasus Selesai (Resolved)', value: stats?.total_resolved || 0, icon: CheckCircle, color: '#06B6D4', bg: 'rgba(6,182,212,0.15)' },
  ];

  const categories = stats?.category_stats ? Object.entries(stats.category_stats) : [];

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Dashboard LostLink Kampus</h1>
          <p className="page-desc">Sistem Cerdas Pemrosesan Bahasa Alami untuk Pencarian & Rekonsiliasi Barang Hilang</p>
        </div>
        <div style={{ display: 'flex', gap: 12 }}>
          <button className="btn btn-secondary" onClick={() => onNavigate('matching')}>
            <Sparkles size={16} color="#818CF8" />
            AI Match Finder
          </button>
          <button className="btn btn-primary" onClick={() => onNavigate('new-report')}>
            + Lapor Barang
          </button>
        </div>
      </div>

      {/* Metric Cards */}
      <div className="stats-grid">
        {statItems.map((item, idx) => {
          const Icon = item.icon;
          return (
            <div key={idx} className="stat-card">
              <div className="stat-icon" style={{ background: item.bg, color: item.color }}>
                <Icon size={24} />
              </div>
              <div>
                <div className="stat-num">{item.value}</div>
                <div className="stat-label">{item.label}</div>
              </div>
            </div>
          );
        })}
      </div>

      {/* NLP Banner */}
      <div className="glass-card" style={{ marginBottom: 28, borderLeft: '4px solid #6366F1', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <div style={{ width: 44, height: 44, borderRadius: 12, background: 'rgba(99,102,241,0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#818CF8' }}>
            <Cpu size={24} />
          </div>
          <div>
            <h3 style={{ fontSize: 16, fontWeight: 700, color: '#fff', marginBottom: 2 }}>
              NLP Model Evaluated: TF-IDF (92.5%) vs BiLSTM (88.7%)
            </h3>
            <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>
              Tersedia notebook mandiri siap dijalankan di Google Colab untuk tugas kuliah dan presentasi akademik.
            </p>
          </div>
        </div>
        <button className="btn btn-secondary" onClick={() => onNavigate('nlp-lab')}>
          Buka NLP Lab <ArrowRight size={14} />
        </button>
      </div>

      {/* Two Column Section */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: 24 }}>
        
        {/* Category Breakdown */}
        <div className="glass-card">
          <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 18, display: 'flex', alignItems: 'center', gap: 8 }}>
            <Tag size={18} color="#06B6D4" />
            Distribusi Kategori Barang
          </h2>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {categories.slice(0, 6).map(([catName, counts], idx) => {
              const total = (counts.lost || 0) + (counts.found || 0);
              const pct = Math.min(Math.round((total / (stats?.total_reports || 1)) * 100 * 3), 100);
              return (
                <div key={idx}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, marginBottom: 4 }}>
                    <span style={{ fontWeight: 600 }}>{catName}</span>
                    <span style={{ color: 'var(--text-muted)' }}>
                      <span style={{ color: '#FB7185' }}>{counts.lost || 0} Hilang</span> • <span style={{ color: '#34D399' }}>{counts.found || 0} Temuan</span>
                    </span>
                  </div>
                  <div style={{ width: '100%', height: 7, background: 'rgba(255,255,255,0.06)', borderRadius: 10, overflow: 'hidden' }}>
                    <div style={{ width: `${pct}%`, height: '100%', background: 'linear-gradient(90deg, #6366F1, #06B6D4)', borderRadius: 10 }}></div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Recent Activity */}
        <div className="glass-card">
          <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 18, display: 'flex', alignItems: 'center', gap: 8 }}>
            <Clock size={18} color="#F59E0B" />
            Aktivitas Pelaporan Terkini
          </h2>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {(stats?.recent_activity || []).map((item, idx) => (
              <div key={idx} style={{ 
                padding: '12px 14px', 
                background: 'rgba(255,255,255,0.02)', 
                border: '1px solid var(--border-color)',
                borderRadius: 'var(--radius-md)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between'
              }}>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 3 }}>
                    <span className={`badge ${item.report_type === 'lost' ? 'badge-lost' : 'badge-found'}`}>
                      {item.report_type === 'lost' ? 'Hilang' : 'Temuan'}
                    </span>
                    <span style={{ fontWeight: 600, fontSize: 14 }}>{item.item_name}</span>
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--text-dim)', display: 'flex', alignItems: 'center', gap: 6 }}>
                    <MapPin size={12} /> {item.location || 'Kampus'} • <Tag size={12} /> {item.category}
                  </div>
                </div>
                <button 
                  style={{ background: 'transparent', border: 'none', color: '#818CF8', cursor: 'pointer', fontSize: 12, fontWeight: 600 }}
                  onClick={() => onNavigate('reports')}
                >
                  Lihat
                </button>
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}
