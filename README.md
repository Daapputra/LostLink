# LostLink

LostLink adalah aplikasi pencarian dan pencocokan laporan barang hilang (*lost*) dan barang temuan (*found*) di lingkungan kampus berbasis Natural Language Processing (NLP). Proyek ini dikembangkan sebagai tugas kuliah NLP dengan fokus membandingkan pendekatan representasi teks klasik (TF-IDF + Logistic Regression) dan Deep Learning (Word Embedding + BiLSTM), serta implementasi sistem rekomendasi pencocokan berbasis Cosine Similarity.

---

## Fitur Utama

1. **Klasifikasi Kategori Otomatis**
   Mengidentifikasi kategori barang (Dompet, Tas, Handphone, Kunci, Kartu Identitas, Laptop, Aksesori, Alat Tulis, Barang Lainnya) dari teks deskripsi bebas laporan.

2. **Pencocokan Laporan (Matching Engine)**
   Mencari dan memeringkat kandidat laporan penemuan yang relevan untuk setiap laporan kehilangan menggunakan kombinasi Cosine Similarity teks (50%), kesesuaian kategori (20%), kesamaan warna (15%), dan kedekatan lokasi (15%).

3. **Dashboard & Web Interface**
   Antarmuka web interaktif untuk pelaporan barang, eksplorasi data, pengujian live NLP sandbox, dan verifikasi kecocokan laporan.

4. **Notebook Eksperimen Google Colab**
   Notebook mandiri yang mencakup alur lengkap dari pengumpulan dataset, preprocessing bahasa Indonesia, pelatihan model, evaluasi metrik, hingga pengujian out-of-distribution.

---

## Struktur Direktori

```text
LostLink/
├── backend/
│   ├── app/
│   │   ├── main.py              # Entrypoint FastAPI
│   │   ├── database.py          # Setup database SQLite dan CRUD
│   │   ├── models.py            # Pydantic schemas
│   │   ├── routes/
│   │   │   ├── reports.py       # API laporan barang
│   │   │   ├── matches.py       # API pencocokan kemiripan
│   │   │   ├── predict.py       # API inferensi klasifikasi kategori
│   │   │   └── stats.py         # API ringkasan statistik
│   │   └── nlp/
│   │       ├── preprocessing.py # Pembersihan teks bahasa Indonesia
│   │       ├── tfidf_model.py   # Wrapper model TF-IDF
│   │       └── dl_model.py      # Wrapper model LSTM
│   ├── trained_models/          # File artefak model tersimpan
│   ├── train_models.py          # Script pelatihan model offline
│   └── requirements.txt         # Dependensi Python
├── frontend/
│   ├── src/
│   │   ├── components/          # Komponen UI React
│   │   ├── pages/               # Halaman Dashboard, Laporan, Matching, NLP Lab
│   │   ├── services/api.js      # Client pemanggilan API backend
│   │   ├── App.jsx
│   │   └── index.css
│   └── package.json
├── dataset/
│   ├── lostlink_dataset.csv     # Dataset 400 laporan sintetis kampus
│   └── generate_dataset.py      # Script generator dataset
├── notebooks/
│   └── LostLink_NLP_Eksperimen_Colab.ipynb # Notebook eksperimen utama
├── docs/
│   └── laporan_akademik.md      # Laporan format akademik lengkap
└── README.md
```

---

## Menjalankan di Google Colab

Eksperimen NLP dapat dijalankan langsung di Google Colab tanpa instalasi lokal:

1. Buka [Google Colab](https://colab.research.google.com/).
2. Unggah file `notebooks/LostLink_NLP_Eksperimen_Colab.ipynb`.
3. Jalankan menu `Runtime > Run all`.
4. Seluruh proses pembuatan dataset, preprocessing, training TF-IDF, training BiLSTM, evaluasi metrik, dan pengujian interaktif akan dieksekusi secara berurutan.

---

## Menjalankan Secara Lokal

### Prasyarat
- Python 3.10 atau lebih baru
- Node.js 18 atau lebih baru dan npm

### 1. Menjalankan Backend (FastAPI)
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```
Dokumentasi interaktif Swagger API dapat diakses di: `http://localhost:8000/docs`

### 2. Menjalankan Frontend (React + Vite)
Buka terminal baru:
```bash
cd frontend
npm install
npm run dev
```
Buka browser di: `http://localhost:5173`

---

## Ringkasan Hasil Evaluasi

Model dievaluasi menggunakan *5-Fold Stratified Group Split* pada dataset 400 sampel kampus untuk mencegah kebocoran informasi (*data leakage*) antara pasangan laporan:

| Metrik Evaluasi | TF-IDF + Logistic Regression | Word Embedding + BiLSTM |
| :--- | :---: | :---: |
| Akurasi Testing | 97.50% | ~90.00% |
| Weighted Precision | 97.75% | ~90.50% |
| Weighted Recall | 97.50% | ~90.00% |
| Weighted F1-Score | 97.49% | ~89.80% |
| Macro F1-Score | 97.44% | ~89.50% |
| Waktu Pelatihan | ~0.11 detik | ~15.2 detik |
| Latensi Prediksi | 0.006 ms/sampel | 12.4 ms/sampel |

### Evaluasi Matching Engine pada Pasangan Ground Truth (80 Pasang):
- **Hit Rate @ Top-1:** 81.25% (65/80)
- **Hit Rate @ Top-3:** 97.50% (78/80)
- **Hit Rate @ Top-5:** 98.75% (79/80)
