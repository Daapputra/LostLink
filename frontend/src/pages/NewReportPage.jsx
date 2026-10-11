import React, { useState, useEffect } from 'react';
import { 
  Sparkles, 
  Send, 
  HelpCircle, 
  CheckCircle, 
  MapPin, 
  Palette, 
  Calendar, 
  Info,
  Check
} from 'lucide-react';
import { api } from '../services/api';

export default function NewReportPage({ onSuccess, onNavigate }) {
  const [reportType, setReportType] = useState('lost');
  const [itemName, setItemName] = useState('');
  const [category, setCategory] = useState('Dompet');
  const [description, setDescription] = useState('');
  const [color, setColor] = useState('');
  const [location, setLocation] = useState('');
  const [eventDate, setEventDate] = useState(new Date().toISOString().split('T')[0]);
  const [contact, setContact] = useState('');
  
  // AI Auto-detect state
  const [aiDetectedCategory, setAiDetectedCategory] = useState('');
  const [aiConfidence, setAiConfidence] = useState(0);
  const [isDetecting, setIsDetecting] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [successMessage, setSuccessMessage] = useState(null);

  const categories = [
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

  // Debounced auto category prediction
  useEffect(() => {
    if (!description || description.trim().length < 8) {
      setAiDetectedCategory('');
      return;
    }

    const timer = setTimeout(async () => {
      setIsDetecting(true);
      try {
        const res = await api.predictCategory(description);
        if (res?.predicted_category) {
          setAiDetectedCategory(res.predicted_category);
          setAiConfidence(Math.round((res.confidence || 0.85) * 100));
        }
      } catch (e) {
        console.error(e);
      } finally {
        setIsDetecting(false);
      }
    }, 600);

    return () => clearTimeout(timer);
  }, [description]);

  const applyAiCategory = () => {
    if (aiDetectedCategory) {
      setCategory(aiDetectedCategory);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!itemName || !description) return;

    setSubmitting(true);
    const newReport = {
      report_type: reportType,
      item_name: itemName,
      category: category,
      description: description,
      color: color,
      location: location,
      event_date: eventDate,
      contact: contact
    };

    const res = await api.createReport(newReport);
    setSubmitting(false);

    setSuccessMessage({
      id: res.id,
      name: itemName,
      type: reportType
    });
  };

  if (successMessage) {
    return (
      <div className="glass-card" style={{ maxWidth: 600, margin: '40px auto', textAlign: 'center', padding: 40 }}>
        <div style={{ width: 64, height: 64, borderRadius: '50%', background: 'rgba(16,185,129,0.2)', color: '#10B981', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 20px auto' }}>
          <Check size={36} />
        </div>
        <h2 style={{ fontSize: 24, fontWeight: 700, marginBottom: 10 }}>Laporan Berhasil Dibuat!</h2>
        <p style={{ color: 'var(--text-muted)', marginBottom: 24, fontSize: 14 }}>
          Laporan <strong>{successMessage.name}</strong> (#{successMessage.id}) telah didaftarkan ke sistem dan siap dicocokkan oleh Agentic NLP Matcher.
        </p>
        <div style={{ display: 'flex', gap: 12, justifyContent: 'center' }}>
          <button className="btn btn-secondary" onClick={() => { setSuccessMessage(null); setItemName(''); setDescription(''); }}>
            + Buat Laporan Lagi
          </button>
          <button className="btn btn-primary" onClick={() => onNavigate('matching')}>
            <Sparkles size={16} /> Cari Kecocokan Sekarang
          </button>
        </div>
      </div>
    );
  }

  return (
    <div style={{ maxWidth: 760, margin: '0 auto' }}>
      <div className="page-header">
        <div>
          <h1 className="page-title">Buat Laporan Baru</h1>
          <p className="page-desc">Isi informasi barang hilang atau temuan di lingkungan kampus</p>
        </div>
      </div>

      <div className="glass-card">
        <form onSubmit={handleSubmit}>
          
          {/* Report Type Selector */}
          <div className="form-group">
            <label className="form-label">Jenis Laporan</label>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
              <button
                type="button"
                onClick={() => setReportType('lost')}
                style={{
                  padding: 16,
                  borderRadius: 'var(--radius-md)',
                  border: reportType === 'lost' ? '2px solid #F43F5E' : '1px solid var(--border-color)',
                  background: reportType === 'lost' ? 'rgba(244,63,94,0.12)' : 'rgba(15,23,42,0.6)',
                  color: reportType === 'lost' ? '#FB7185' : 'var(--text-muted)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 12,
                  fontWeight: 700,
                  fontSize: 15,
                  cursor: 'pointer'
                }}
              >
                <HelpCircle size={22} />
                <div style={{ textAlign: 'left' }}>
                  <div>Saya Kehilangan Barang</div>
                  <div style={{ fontSize: 11, fontWeight: 400, opacity: 0.8 }}>Laporkan barang hilang milik Anda</div>
                </div>
              </button>

              <button
                type="button"
                onClick={() => setReportType('found')}
                style={{
                  padding: 16,
                  borderRadius: 'var(--radius-md)',
                  border: reportType === 'found' ? '2px solid #10B981' : '1px solid var(--border-color)',
                  background: reportType === 'found' ? 'rgba(16,185,129,0.12)' : 'rgba(15,23,42,0.6)',
                  color: reportType === 'found' ? '#34D399' : 'var(--text-muted)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 12,
                  fontWeight: 700,
                  fontSize: 15,
                  cursor: 'pointer'
                }}
              >
                <CheckCircle size={22} />
                <div style={{ textAlign: 'left' }}>
                  <div>Saya Menemukan Barang</div>
                  <div style={{ fontSize: 11, fontWeight: 400, opacity: 0.8 }}>Laporkan barang tak bertuan yang Anda temukan</div>
                </div>
              </button>
            </div>
          </div>

          {/* Item Name */}
          <div className="form-group">
            <label className="form-label">Nama Barang *</label>
            <input 
              type="text"
              required
              placeholder="Contoh: Dompet Lipat Hitam, KTM Mahasiswa, Laptop Asus ROG..."
              value={itemName}
              onChange={(e) => setItemName(e.target.value)}
              className="form-input"
            />
          </div>

          {/* Description */}
          <div className="form-group">
            <label className="form-label">Deskripsi Lengkap & Ciri-Ciri Khusus *</label>
            <textarea 
              required
              rows={4}
              placeholder="Tuliskan deskripsi detail: bentuk, isi di dalamnya, stiker, goresan, kondisi saat hilang/ditemukan..."
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="form-textarea"
            />

            {/* AI Auto-Detection Hint */}
            {isDetecting && (
              <div style={{ fontSize: 12, color: '#818CF8', marginTop: 6, display: 'flex', alignItems: 'center', gap: 6 }}>
                <Sparkles size={14} className="animate-spin" /> NLP Agent sedang menganalisis deskripsi...
              </div>
            )}

            {aiDetectedCategory && (
              <div className="ai-helper-box">
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <Sparkles size={18} color="#818CF8" />
                  <span style={{ fontSize: 13 }}>
                    AI Merekomendasikan Kategori: <strong style={{ color: '#fff' }}>{aiDetectedCategory}</strong> ({aiConfidence}% confidence)
                  </span>
                </div>
                {category !== aiDetectedCategory && (
                  <button
                    type="button"
                    onClick={applyAiCategory}
                    className="btn btn-secondary"
                    style={{ padding: '4px 12px', fontSize: 12, color: '#818CF8' }}
                  >
                    Gunakan Kategori Ini
                  </button>
                )}
              </div>
            )}
          </div>

          {/* Category & Color Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <div className="form-group">
              <label className="form-label">Kategori Barang</label>
              <select 
                value={category}
                onChange={(e) => setCategory(e.target.value)}
                className="form-select"
              >
                {categories.map((c, i) => (
                  <option key={i} value={c}>{c}</option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Warna Utama</label>
              <input 
                type="text"
                placeholder="Contoh: Hitam, Biru Navy, Coklat..."
                value={color}
                onChange={(e) => setColor(e.target.value)}
                className="form-input"
              />
            </div>
          </div>

          {/* Location & Date Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <div className="form-group">
              <label className="form-label">Lokasi Kejadian di Kampus</label>
              <input 
                type="text"
                placeholder="Contoh: Perpustakaan Lt 2, Mushola Gedung C, Kantin..."
                value={location}
                onChange={(e) => setLocation(e.target.value)}
                className="form-input"
              />
            </div>

            <div className="form-group">
              <label className="form-label">Tanggal Kejadian</label>
              <input 
                type="date"
                value={eventDate}
                onChange={(e) => setEventDate(e.target.value)}
                className="form-input"
              />
            </div>
          </div>

          {/* Contact */}
          <div className="form-group">
            <label className="form-label">Kontak / No. WhatsApp (Opsional)</label>
            <input 
              type="text"
              placeholder="Contoh: 08123456789 (Hanya untuk verifikasi)"
              value={contact}
              onChange={(e) => setContact(e.target.value)}
              className="form-input"
            />
          </div>

          <div style={{ marginTop: 24, display: 'flex', justifyContent: 'flex-end', gap: 12 }}>
            <button type="button" className="btn btn-secondary" onClick={() => onNavigate('dashboard')}>
              Batal
            </button>
            <button type="submit" disabled={submitting} className="btn btn-primary" style={{ padding: '12px 28px' }}>
              {submitting ? 'Menyimpan...' : (
                <>
                  <Send size={16} /> Publikasikan Laporan
                </>
              )}
            </button>
          </div>

        </form>
      </div>
    </div>
  );
}
