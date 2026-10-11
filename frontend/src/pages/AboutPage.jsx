import React from 'react';
import { 
  Info, 
  Layers, 
  Sparkles, 
  Cpu, 
  Binary, 
  FileText, 
  CheckCircle,
  FolderGit2
} from 'lucide-react';

export default function AboutPage() {
  return (
    <div style={{ maxWidth: 860, margin: '0 auto' }}>
      <div className="page-header">
        <div>
          <h1 className="page-title">Tentang Proyek LostLink</h1>
          <p className="page-desc">AI-Powered Campus Lost & Found Agent — Tugas Mata Kuliah Natural Language Processing</p>
        </div>
      </div>

      <div className="glass-card" style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: 20, fontWeight: 700, color: '#fff', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 10 }}>
          <Layers size={22} color="#6366F1" /> Visi & Konsep "Agentic Campus"
        </h2>
        <p style={{ fontSize: 14, color: 'var(--text-muted)', lineHeight: 1.7, marginBottom: 14 }}>
          <strong>LostLink</strong> dibangun untuk menyelesaikan masalah umum di lingkungan perguruan tinggi: kehilangan barang berharga seperti KTM, dompet, kunci, dan gadget yang sering kali sulit ditemukan kembali karena sistem informasi yang manual dan tersebar di puluhan grup chat.
        </p>
        <p style={{ fontSize: 14, color: 'var(--text-muted)', lineHeight: 1.7 }}>
          Dengan pendekatan <strong>Agentic AI</strong>, sistem ini bertindak secara proaktif sebagai agen yang menganalisis bahasa alami pengguna, mengekstraksi entitas kunci, memprediksi kategori barang, dan mencocokkan laporan kehilangan dengan laporan temuan secara otomatis.
        </p>
      </div>

      <div className="glass-card" style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: 20, fontWeight: 700, color: '#fff', marginBottom: 16, display: 'flex', alignItems: 'center', gap: 10 }}>
          <Cpu size={22} color="#06B6D4" /> Metodologi NLP yang Digunakan
        </h2>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 16 }}>
          
          <div style={{ padding: 16, background: 'rgba(255,255,255,0.02)', borderRadius: 12, border: '1px solid var(--border-color)' }}>
            <h4 style={{ color: '#818CF8', fontSize: 15, fontWeight: 700, marginBottom: 6 }}>1. Preprocessing Khusus</h4>
            <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              Case folding, typo/slang normalization kampus, serta stopword filtering yang mempertahankan kata negasi dan lokasi.
            </p>
          </div>

          <div style={{ padding: 16, background: 'rgba(255,255,255,0.02)', borderRadius: 12, border: '1px solid var(--border-color)' }}>
            <h4 style={{ color: '#34D399', fontSize: 15, fontWeight: 700, marginBottom: 6 }}>2. TF-IDF + Logistic Regression</h4>
            <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              Ekstraksi fitur unigram + bigram berbobot statistik dengan akurasi 92.5% dan latensi inferensi sub-milidetik.
            </p>
          </div>

          <div style={{ padding: 16, background: 'rgba(255,255,255,0.02)', borderRadius: 12, border: '1px solid var(--border-color)' }}>
            <h4 style={{ color: '#FBBF24', fontSize: 15, fontWeight: 700, marginBottom: 6 }}>3. Word Embedding + BiLSTM</h4>
            <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              Pemodelan sekuensial dua arah untuk menangkap konteks kalimat dan relasi dependensi kata yang lebih kaya.
            </p>
          </div>

          <div style={{ padding: 16, background: 'rgba(255,255,255,0.02)', borderRadius: 12, border: '1px solid var(--border-color)' }}>
            <h4 style={{ color: '#F43F5E', fontSize: 15, fontWeight: 700, marginBottom: 6 }}>4. Cosine Similarity Matcher</h4>
            <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              Perhitungan kesamaan multi-fitur terkalibrasi: Deskripsi (50%), Kategori (20%), Warna (15%), dan Lokasi (15%).
            </p>
          </div>

        </div>
      </div>

      <div className="glass-card">
        <h2 style={{ fontSize: 20, fontWeight: 700, color: '#fff', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 10 }}>
          <FolderGit2 size={22} color="#10B981" /> Dokumen & Sumber Daya Proyek
        </h2>
        <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 10, fontSize: 14 }}>
          <li style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <CheckCircle size={16} color="#10B981" />
            <span><strong>Laporan Akademik Lengkap:</strong> <code>docs/laporan_akademik.md</code></span>
          </li>
          <li style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <CheckCircle size={16} color="#10B981" />
            <span><strong>Google Colab Notebook:</strong> <code>notebooks/LostLink_NLP_Eksperimen_Colab.ipynb</code></span>
          </li>
          <li style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <CheckCircle size={16} color="#10B981" />
            <span><strong>FastAPI Backend:</strong> <code>backend/app/main.py</code></span>
          </li>
          <li style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <CheckCircle size={16} color="#10B981" />
            <span><strong>Dataset Sintetis (400 Baris):</strong> <code>dataset/lostlink_dataset.csv</code></span>
          </li>
        </ul>
      </div>

    </div>
  );
}
