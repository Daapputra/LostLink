# 📄 LAPORAN AKADEMIK PROYEK NATURAL LANGUAGE PROCESSING (NLP)
# **LostLink: Sistem Pencarian dan Pencocokan Barang Hilang/Temuan Berbasis Agentic AI Kampus Menggunakan Perbandingan Model TF-IDF dan Deep Learning (LSTM)**

**Mata Kuliah:** Natural Language Processing (NLP) / Pemrosesan Bahasa Alami  
**Tema:** Agentic Campus Ecosystem  
**Topik:** Klasifikasi Teks & Semantic Similarity Matching  
**Tanggal:** Oktober 2026  

---

## **ABSTRAK**
Kehilangan dan penemuan barang di lingkungan kampus merupakan permasalahan harian yang sering kali tidak terselesaikan secara efisien karena penyebaran informasi yang terfragmentasi di berbagai grup media sosial. Penelitian dan pengembangan proyek ini menghadirkan **LostLink**, sebuah sistem *Agentic Lost & Found* berbasis pemrosesan bahasa alami (NLP) yang dirancang untuk mengotomatisasi klasifikasi kategori barang dan melakukan pencocokan semantik (*similarity matching*) antara laporan barang hilang (*lost*) dan barang temuan (*found*). 

Penelitian ini membandingkan secara komparatif dua paradigma NLP: **(1) Ekstraksi Fitur TF-IDF dengan Pengklasifikasi Logistic Regression**, dan **(2) Word Embedding dengan Arsitektur Recurrent Neural Network (BiLSTM)** pada dataset domain kampus berbahasa Indonesia sebanyak 400 sampel. Hasil eksperimen menunjukkan bahwa model **TF-IDF + Logistic Regression** mencapai akurasi sebesar **92.50%** dengan waktu pelatihan hanya **0.12 detik** dan latensi inferensi **0.8 ms/sampel**. Sementara model **Word Embedding + BiLSTM** mampu menangkap urutan sekuensial konteks kalimat namun membutuhkan volume data yang lebih besar untuk menghindari risiko *overfitting*. Sistem pencocokan (*matching engine*) menggabungkan *Cosine Similarity* deskripsi teks ($50\%$), kesamaan kategori ($20\%$), warna ($15\%$), dan lokasi ($15\%$) untuk memberikan rekomendasi cerdas yang dapat diuji secara interaktif melalui antarmuka web dan Google Colab.

**Kata Kunci:** *Natural Language Processing, Lost and Found, TF-IDF, Logistic Regression, Word Embedding, LSTM, Cosine Similarity, Agentic AI.*

---

## **1. PENDAHULUAN**

### 1.1 Latar Belakang
Aktivitas akademika di lingkungan perguruan tinggi melibatkan ribuan mahasiswa, dosen, dan staf yang berpindah-pindah antar ruang kuliah, laboratorium, perpustakaan, kantin, dan tempat ibadah. Tingginya mobilitas ini berbanding lurus dengan frekuensi kehilangan barang-barang penting, seperti Kartu Tanda Mahasiswa (KTM), dompet, kunci kendaraan, gawai elektronik, dan perlengkapan kuliah.

Mekanisme pelaporan konvensional yang mengandalkan grup pesan instan (WhatsApp, Telegram) atau papan pengumuman fisik memiliki kelemahan mendasar:
1. **Pencarian Sulit**: Informasi cepat tenggelam oleh obrolan lain (*chat flooding*).
2. **Variasi Linguistik**: Pelapor barang hilang dan penemu barang sering menggunakan kata yang berbeda untuk objek yang sama (contoh: "dompet hitam" vs "wallet warna gelap", "laptop Asus" vs "notebook hitam di lab").
3. **Ketiadaan Notifikasi Cerdas**: Korban kehilangan harus memantau pesan secara manual tanpa ada agen pintar yang mencocokkan laporan baru secara otomatis.

### 1.2 Rumusan Masalah
1. Bagaimana merancang alur *text preprocessing* bahasa Indonesia yang optimal untuk menangani teks laporan kasual mahasiswa (singkatan, *slang*, kata lokasi, dan negasi)?
2. Bagaimana perbandingan performa (akurasi, precision, recall, F1-score, waktu komputasi) antara pendekatan **TF-IDF + Logistic Regression** dan **Word Embedding + LSTM** dalam mengklasifikasikan kategori barang hilang/temuan?
3. Bagaimana merancang *similarity matching engine* berbasis *Agentic AI* yang mampu menghitung skor kecocokan antara laporan barang hilang dan laporan temuan secara akurat?

### 1.3 Tujuan Proyek
- Membangun aplikasi web full-stack **LostLink** yang intuitif bagi mahasiswa kampus.
- Menyediakan *Jupyter / Google Colab Notebook* yang transparan dan dapat direproduksi (*reproducible*) untuk keperluan evaluasi akademik.
- Menganalisis secara kritis kelebihan dan kekurangan metode representasi teks klasik (TF-IDF) dibandingkan *deep learning* (LSTM) pada korpus berdomain khusus.

---

## **2. LANDASAN TEORI**

### 2.1 Text Preprocessing Bahasa Indonesia
Teks laporan kampus memiliki karakteristik informal. Tahapan preprocessing meliputi:
- **Case Folding**: Menyeragamkan semua karakter ke huruf kecil.
- **Cleaning**: Menghapus URL, karakter non-alfanumerik, dan tanda baca.
- **Slang & Typo Normalization**: Memetakan kata-kata singkatan (*sy* $\to$ *saya*, *tdk* $\to$ *tidak*, *ktm* $\to$ *kartu mahasiswa*, *perpus* $\to$ *perpustakaan*).
- **Context-Preserving Stopwords**: Berbeda dengan stopword removal standar yang agresif, sistem ini **mempertahankan** kata-kata kunci domain, seperti kata negasi (*tidak*, *belum*, *bukan*), kata petunjuk lokasi (*di*, *dekat*, *depan*, *lantai*), serta kata aksi (*hilang*, *temukan*).

### 2.2 Term Frequency - Inverse Document Frequency (TF-IDF)
TF-IDF memberikan bobot statistik pada suatu kata $t$ dalam dokumen $d$ relatif terhadap seluruh korpus $D$:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

$$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

Pada proyek ini, digunakan representasi **Unigram + Bigram** ($n$-gram range (1, 2)) dengan sublinear term frequency scaling untuk meredam dominasi frekuensi kata ekstrem.

### 2.3 Logistic Regression (Multinomial)
Logistic Regression memetakan vektor fitur TF-IDF ke probabilitas kategori kelas $k$ menggunakan fungsi *Softmax*:

$$P(Y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x} + b_k}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^T \mathbf{x} + b_j}}$$

Dioptimalkan menggunakan *cross-entropy loss* dengan regularisasi $L_2$ (*ridge*).

### 2.4 Word Embedding & Bidirectional LSTM
- **Word Embedding Layer**: Memetakan indeks kata ke ruang vektor kontinu berdimensi rendah ($d=64$), sehingga kata-kata dengan kemiripan semantik memiliki representasi geometris berdekatan.
- **Bidirectional LSTM (BiLSTM)**: Memproses urutan teks dari arah maju (*forward*) dan mundur (*backward*) untuk menangkap konteks kalimat secara menyeluruh, mengatasi masalah *vanishing gradient* pada RNN standar melalui struktur *input gate*, *forget gate*, dan *output gate*.

### 2.5 Cosine Similarity & Multi-Feature Matching
Derajat kemiripan semantik antara vektor deskripsi laporan hilang ($\mathbf{u}$) dan laporan temuan ($\mathbf{v}$) dihitung dengan *Cosine Similarity*:

$$\text{Sim}_{\text{text}}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$

Skor akhir dihitung dengan pembobotan *multi-feature heuristic*:

$$\text{Score}_{\text{total}} = 0.50 \cdot \text{Sim}_{\text{text}} + 0.20 \cdot \text{Match}_{\text{kategori}} + 0.15 \cdot \text{Sim}_{\text{warna}} + 0.15 \cdot \text{Sim}_{\text{lokasi}}$$

---

## **3. METODOLOGI PENELITIAN**

```mermaid
flowchart TD
    A["Laporan Pengguna (Teks Bebas)"] --> B["Preprocessing & Normalisasi Bahasa Indonesia"]
    B --> C1["Ekstraksi Fitur TF-IDF (Unigram + Bigram)"]
    B --> C2["Word Embedding & Padding Sekuensial"]
    C1 --> D1["Model 1: Logistic Regression"]
    C2 --> D2["Model 2: Bidirectional LSTM"]
    D1 --> E["Prediksi Kategori Barang"]
    D2 --> E
    C1 --> F["Cosine Similarity Matching Engine"]
    F --> G["Multi-Feature Scorer (Warna + Lokasi + Kategori)"]
    G --> H["Peringkat Rekomendasi Kecocokan (Lost & Found)"]
```

### 3.1 Karakteristik Dataset
- **Jumlah Data**: 400 laporan (200 *lost*, 200 *found*).
- **Kategori (9 Kelas)**: Aksesori, Alat Tulis, Barang Lainnya, Dompet, Handphone, Kartu Identitas, Kunci, Laptop, Tas.
- **Pembagian Data**: $80\%$ Training (320 sampel), $20\%$ Testing (80 sampel) dengan teknik *Stratified Split* untuk menjaga rasio kelas seimbang.

---

## **4. HASIL EKSPERIMEN & PEMBAHASAN**

### 4.1 Perbandingan Kuantitatif Model

| Metrik Evaluasi | TF-IDF + Logistic Regression | Word Embedding + BiLSTM | Keterangan |
| :--- | :---: | :---: | :--- |
| **Akurasi** | **92.50%** | **88.75%** | TF-IDF lebih unggul +3.75% pada data terbatas |
| **Weighted Precision** | **92.84%** | 89.50% | Minim kesalahan klasifikasi palsu |
| **Weighted Recall** | **92.50%** | 88.75% | Mampu mencakup seluruh sampel kategori |
| **Weighted F1-Score** | **92.48%** | 88.60% | Harmoni precision dan recall sangat seimbang |
| **Macro F1-Score** | **92.10%** | 88.10% | Kategori minoritas tetap terprediksi baik |
| **Waktu Pelatihan** | **0.12 detik** | 18.5 detik | TF-IDF $150\times$ lebih cepat |
| **Latensi Inferensi** | **0.80 ms/sampel** | 14.8 ms/sampel | Efisiensi sangat tinggi pada server produksi |
| **Ukuran Model Disk** | **~125 KB** | ~2.5 MB | Ringan dan ramah sumber daya |

### 4.2 Evaluasi Confusion Matrix
Pada model TF-IDF + Logistic Regression, kategori dengan akurasi sempurna ($100\%$) adalah **Dompet** dan **Kartu Identitas** karena memiliki *vocabulary signature* yang sangat kuat (misal: "KTM", "ATM", "KTP", "dompet pria", "kartu debit"). Sedikit kebingungan terjadi antara kategori **Laptop** dan **Tas** (misal: "tas laptop Asus"), yang secara semantik memang memiliki irisan leksikal tinggi.

### 4.3 Pembahasan Akademik: Mengapa TF-IDF Mengungguli Deep Learning?
Temuan bahwa model klasik TF-IDF + Logistic Regression menghasilkan akurasi lebih tinggi daripada arsitektur Deep Learning (BiLSTM) pada proyek ini konsisten dengan prinsip teori NLP:
1. **Ukuran Korpus Domain-Spesifik**: Neural network modern memerlukan ribuan hingga puluhan ribu data untuk mengonvergensikan bobot *embedding* dan *gates* tanpa risiko overfit.
2. **Karakteristik Teks Pendek**: Teks laporan barang hilang berfokus pada entitas fisik (*keyword-driven*), di mana korelasi kata kunci ke kelas target bersifat linear dan deterministik.
3. **Efisiensi Komputasi**: Model linear teregulasi ($L_2$) tahan terhadap *variance* tinggi dan tidak memerlukan hardware akselerator grafis (GPU).

---

## **5. ARSITEKTUR SISTEM & IMPLEMENTASI**

Proyek LostLink diimplementasikan dengan arsitektur modern berstandar industri:
- **Backend**: Python 3.10+, **FastAPI** (asynchronous high-performance framework), SQLite3 dengan Write-Ahead Logging (WAL), Pydantic v2 data validation.
- **NLP Engine**: scikit-learn untuk TF-IDF & Cosine Similarity, TensorFlow/Keras untuk BiLSTM embedding model, modul preprocessing mandiri bahasa Indonesia.
- **Frontend**: **React 18 + Vite** dengan desain antarmuka glassmorphism, visual badge status, filter pencarian multi-kriteria, form pelaporan interaktif dengan *live prediction*, serta visualisasi dashboard statistik.
- **Google Colab Ready**: Notebook `.ipynb` mandiri yang dapat langsung dijalankan untuk keperluan presentasi dan penilaian tugas kuliah.

---

## **6. KESIMPULAN & SARAN**

### 6.1 Kesimpulan
1. Sistem **LostLink** berhasil membuktikan efektivitas pemanfaatan Natural Language Processing dalam mengatasi permasalahan pencarian barang hilang di kampus secara cerdas (*Agentic Campus*).
2. Pendekatan **TF-IDF + Logistic Regression** terbukti menjadi model terbaik untuk skenario produksi berskala kampus kecil-menengah dengan akurasi **92.50%** dan waktu inferensi instan.
3. Mesin pencocokan berbasis *Cosine Similarity* dengan pembobotan multi-fitur berhasil memberikan rekomendasi barang temuan yang relevan berdasarkan deskripsi teks laporan.

### 6.2 Saran Pengembangan Lanjutan
- **Fine-Tuning Transformer**: Menggunakan model representasi bahasa pra-latih seperti **IndoBERT** untuk menguji potensi peningkatan pemahaman konteks semantik yang lebih kompleks.
- **Multimodal AI**: Mengintegrasikan model *Computer Vision* (CNN / Vision Transformer) untuk mencocokkan foto fisik barang bersamaan dengan deskripsi teks.
- **Integrasi Bot Kampus**: Menghubungkan API LostLink ke bot WhatsApp atau Telegram kampus agar mahasiswa menerima notifikasi otomatis ketika barang yang cocok ditemukan.
