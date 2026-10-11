"""
LostLink Text Preprocessing Module
===================================
Modul preprocessing teks Bahasa Indonesia untuk sistem LostLink.

Tahapan preprocessing:
1. Mengubah teks menjadi lowercase
2. Menghapus URL dan email
3. Menghapus karakter dan simbol tidak relevan
4. Membersihkan spasi berlebih
5. Opsional: menghapus stopwords secara selektif
6. Menangani nilai kosong

CATATAN:
- Stopwords TIDAK dihapus secara membabi buta.
- Kata negasi (tidak, bukan, belum) dan konteks lokasi (di, dekat, sekitar)
  tetap dipertahankan karena penting untuk makna laporan.
"""

import re


# ============================================================
# STOPWORDS BAHASA INDONESIA (SELEKTIF)
# ============================================================

# Stopwords yang akan dihapus
STOPWORDS_REMOVE = {
    'yang', 'dan', 'ini', 'itu', 'dengan', 'untuk', 'pada',
    'adalah', 'akan', 'bisa', 'ada', 'juga', 'sudah', 'telah',
    'oleh', 'saat', 'secara', 'sebagai', 'serta', 'hal', 'dapat',
    'tersebut', 'hanya', 'sebuah', 'agar', 'bagi', 'antara',
    'seperti', 'namun', 'ketika', 'bahwa', 'atau', 'lebih',
    'karena', 'yaitu', 'lalu', 'ia', 'kami', 'mereka', 'kita',
    'para', 'pun', 'maupun', 'apakah', 'begitu', 'maka',
    'sedangkan', 'sehingga', 'terhadap', 'tentang', 'sambil',
    'supaya', 'selama', 'setelah', 'sebelum', 'sering', 'pernah',
    'sangat', 'terlalu', 'hampir', 'cukup', 'sudah', 'masih',
}

# Kata-kata yang HARUS dipertahankan (penting untuk konteks lost & found)
KEEP_WORDS = {
    # Negasi
    'tidak', 'bukan', 'belum', 'tanpa', 'jangan',
    # Lokasi & posisi
    'di', 'ke', 'dari', 'dekat', 'sekitar', 'depan', 'belakang',
    'atas', 'bawah', 'samping', 'dalam', 'luar', 'area', 'antara',
    # Kata kunci lost & found
    'hilang', 'temukan', 'kehilangan', 'menemukan', 'ditemukan',
    'tertinggal', 'terjatuh', 'lupa', 'cari', 'mencari',
    # Kepemilikan
    'saya', 'milik', 'punya',
    # Warna & deskripsi
    'warna', 'berwarna', 'berisi', 'berbentuk',
    # Waktu
    'kemarin', 'tadi', 'pagi', 'siang', 'sore', 'malam',
}


def preprocess_text(text, remove_stopwords=False):
    """
    Preprocessing teks Bahasa Indonesia.

    Tahapan:
    1. Lowercase
    2. Hapus URL dan email
    3. Hapus karakter/simbol tidak relevan
    4. Hapus angka yang berdiri sendiri
    5. Bersihkan spasi berlebih
    6. Opsional: hapus stopwords secara selektif

    Args:
        text (str): Teks input yang akan dipreprocess.
        remove_stopwords (bool): Jika True, hapus stopwords secara selektif.
            Default False karena beberapa model (embedding) membutuhkan konteks penuh.

    Returns:
        str: Teks yang sudah dipreprocess. Mengembalikan string kosong jika input invalid.

    Contoh:
        >>> preprocess_text("Saya Kehilangan DOMPET di Perpustakaan!!!")
        'saya kehilangan dompet di perpustakaan'
        >>> preprocess_text("HP saya hilang!!!", remove_stopwords=True)
        'hp saya hilang'
    """
    # Tangani nilai kosong / non-string
    if not text or not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Hapus URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # 3. Hapus email
    text = re.sub(r'\S+@\S+\.\S+', '', text)

    # 4. Hapus nomor telepon (format Indonesia)
    text = re.sub(r'(\+62|62|0)\d{8,12}', '', text)

    # 5. Hapus karakter khusus, pertahankan huruf, angka, spasi, dan tanda hubung
    text = re.sub(r'[^\w\s-]', ' ', text)

    # 6. Hapus underscore
    text = text.replace('_', ' ')

    # 7. Hapus angka yang berdiri sendiri (tapi pertahankan angka dalam kata, misal "A201")
    text = re.sub(r'\b\d+\b', '', text)

    # 8. Bersihkan spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()

    # 9. Opsional: hapus stopwords secara selektif
    if remove_stopwords:
        words = text.split()
        words = [w for w in words if w not in STOPWORDS_REMOVE or w in KEEP_WORDS]
        text = ' '.join(words)

    return text


def preprocess_for_tfidf(text):
    """
    Preprocessing khusus untuk pipeline TF-IDF.
    Menghapus stopwords secara selektif untuk meningkatkan kualitas fitur TF-IDF.

    Args:
        text (str): Teks input.

    Returns:
        str: Teks yang sudah dipreprocess untuk TF-IDF.
    """
    return preprocess_text(text, remove_stopwords=True)


def preprocess_for_embedding(text):
    """
    Preprocessing khusus untuk pipeline Word Embedding.
    Tidak menghapus stopwords karena embedding layer akan mempelajari
    representasi setiap kata dari konteks.

    Args:
        text (str): Teks input.

    Returns:
        str: Teks yang sudah dipreprocess untuk embedding.
    """
    return preprocess_text(text, remove_stopwords=False)


def preprocess_dataframe(df, text_column='description', method='general'):
    """
    Preprocess seluruh kolom teks dalam DataFrame.

    Args:
        df (pd.DataFrame): DataFrame input.
        text_column (str): Nama kolom yang berisi teks.
        method (str): Metode preprocessing - 'general', 'tfidf', atau 'embedding'.

    Returns:
        pd.DataFrame: DataFrame dengan kolom baru '{text_column}_clean'.

    Side effects:
        - Mengisi nilai kosong pada text_column dengan string kosong.
        - Menghapus baris duplikat berdasarkan text_column.
        - Mencetak jumlah duplikat yang dihapus (jika ada).
    """
    df = df.copy()

    # Tangani missing values
    df[text_column] = df[text_column].fillna('')

    # Pilih fungsi preprocessing
    if method == 'tfidf':
        preprocess_fn = preprocess_for_tfidf
    elif method == 'embedding':
        preprocess_fn = preprocess_for_embedding
    else:
        preprocess_fn = preprocess_text

    # Terapkan preprocessing
    clean_col = f'{text_column}_clean'
    df[clean_col] = df[text_column].apply(preprocess_fn)

    # Periksa dan hapus duplikasi
    initial_len = len(df)
    df = df.drop_duplicates(subset=[text_column], keep='first')
    removed = initial_len - len(df)
    if removed > 0:
        print(f"[Preprocessing] Menghapus {removed} baris duplikat")

    # Hapus baris dengan teks kosong setelah preprocessing
    empty_count = (df[clean_col] == '').sum()
    if empty_count > 0:
        print(f"[Preprocessing] Ditemukan {empty_count} baris dengan teks kosong setelah preprocessing")

    return df


def show_preprocessing_examples(texts, method='general'):
    """
    Menampilkan contoh teks sebelum dan sesudah preprocessing.
    Berguna untuk dokumentasi dan debugging.

    Args:
        texts (list): Daftar teks contoh.
        method (str): Metode preprocessing.

    Returns:
        list: Daftar tuple (teks_asli, teks_clean).
    """
    if method == 'tfidf':
        preprocess_fn = preprocess_for_tfidf
    elif method == 'embedding':
        preprocess_fn = preprocess_for_embedding
    else:
        preprocess_fn = preprocess_text

    results = []
    print(f"\n{'='*60}")
    print(f"Contoh Preprocessing (metode: {method})")
    print(f"{'='*60}")

    for i, text in enumerate(texts, 1):
        clean = preprocess_fn(text)
        results.append((text, clean))
        print(f"\n[{i}] SEBELUM: {text}")
        print(f"    SESUDAH: {clean}")

    print(f"\n{'='*60}")
    return results


def combine_text_features(row, columns=None):
    """
    Menggabungkan beberapa kolom teks menjadi satu string untuk matching.
    Berguna untuk menggabungkan deskripsi, kategori, warna, dan lokasi.

    Args:
        row (pd.Series): Baris DataFrame.
        columns (list): Daftar nama kolom yang akan digabungkan.
            Default: ['description', 'category', 'color', 'location']

    Returns:
        str: Teks gabungan yang sudah dipreprocess.
    """
    if columns is None:
        columns = ['description', 'category', 'color', 'location']

    parts = []
    for col in columns:
        val = row.get(col, '')
        if val and isinstance(val, str):
            parts.append(val)

    combined = ' '.join(parts)
    return preprocess_text(combined)


# ============================================================
# CONTOH PENGGUNAAN (saat dijalankan langsung)
# ============================================================

if __name__ == '__main__':
    # Contoh penggunaan
    examples = [
        "Saya Kehilangan DOMPET KULIT di Perpustakaan Pusat!!!",
        "Menemukan HP Samsung Galaxy di kantin FT. Hub: 081234567890",
        "Tas ransel warna hitam tertinggal di GKB lt.2, berisi laptop & charger.",
        "KTM atas nama M. Rizki, NIM 2023XXXXXX, email: rizki@univ.ac.id",
        None,
        "",
        "   Kunci   motor   Honda   Beat   di   parkiran   ",
        "TOLONG! Dompet saya yang berwarna COKLAT hilang di masjid kampus!!!"
    ]

    print("=" * 60)
    print("LostLink Preprocessing Module - Demo")
    print("=" * 60)

    show_preprocessing_examples(examples, method='general')
    print()
    show_preprocessing_examples(examples[:3], method='tfidf')
    print()
    show_preprocessing_examples(examples[:3], method='embedding')
