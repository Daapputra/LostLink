import React from 'react';
import { 
  LayoutDashboard, 
  FileText, 
  PlusCircle, 
  Sparkles, 
  Binary, 
  Info,
  Layers
} from 'lucide-react';

export default function Sidebar({ currentTab, setCurrentTab, stats }) {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'reports', label: 'Daftar Laporan', icon: FileText, badge: stats?.total_reports },
    { id: 'new-report', label: 'Buat Laporan', icon: PlusCircle, isHighlight: true },
    { id: 'matching', label: 'AI Match Finder', icon: Sparkles, badge: 'Agentic' },
    { id: 'nlp-lab', label: 'NLP Lab & Colab', icon: Binary },
    { id: 'about', label: 'Tentang Proyek', icon: Info },
  ];

  return (
    <aside className="sidebar">
      <div className="brand-logo">
        <div className="logo-badge">
          <Layers size={22} color="#fff" />
        </div>
        <div>
          <div className="brand-title">LostLink</div>
          <div className="brand-subtitle">AI Campus Agent</div>
        </div>
      </div>

      <ul className="nav-links">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentTab === item.id;
          return (
            <li key={item.id} className={`nav-item ${isActive ? 'active' : ''}`}>
              <button onClick={() => setCurrentTab(item.id)}>
                <Icon size={18} />
                <span>{item.label}</span>
                {item.badge && <span className="nav-badge">{item.badge}</span>}
              </button>
            </li>
          );
        })}
      </ul>

      <div className="sidebar-footer">
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
          <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#10B981', display: 'inline-block' }}></span>
          <span style={{ fontWeight: 600, color: '#E2E8F0', fontSize: 13 }}>FastAPI + TF-IDF</span>
        </div>
        <div>NLP Agentic Campus v1.0</div>
      </div>
    </aside>
  );
}
