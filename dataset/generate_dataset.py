#!/usr/bin/env python3
"""
LostLink Dataset Generator
==========================
Menghasilkan dataset sintetis laporan barang hilang dan ditemukan
di lingkungan kampus dalam Bahasa Indonesia.

CATATAN PENTING:
- Dataset ini bersifat SINTETIS dan dibuat untuk keperluan demonstrasi dan eksperimen NLP.
- Bukan data nyata dari mahasiswa.
- match_group > 0 menandakan pasangan laporan yang merujuk pada barang yang sama.
  match_group = 0 berarti laporan tidak memiliki pasangan yang diketahui.
- Deskripsi pada pasangan dirancang dengan variasi leksikal agar tidak trivial.

Cara menjalankan:
    python generate_dataset.py
"""

import csv
import random
from datetime import datetime, timedelta
from collections import Counter

random.seed(42)

# ============================================================
# KONFIGURASI
# ============================================================
NUM_MATCH_GROUPS = 80       # 80 pasangan (160 laporan berpasangan)
NUM_UNMATCHED = 240         # 240 laporan tanpa pasangan
# Total: 160 + 240 = 400 laporan

CATEGORIES = [
    'Dompet', 'Tas', 'Handphone', 'Kunci', 'Kartu Identitas',
    'Laptop', 'Aksesori', 'Alat Tulis', 'Barang Lainnya'
]

# ============================================================
# DATA TEMPLATE
# ============================================================

ITEMS_PER_CATEGORY = {
    'Dompet': ['dompet kulit', 'dompet lipat', 'dompet kain', 'dompet kecil',
               'dompet panjang', 'dompet pria', 'dompet wanita'],
    'Tas': ['tas ransel', 'tas selempang', 'tas laptop', 'tas jinjing',
            'totebag', 'tas punggung', 'sling bag', 'tas kecil'],
    'Handphone': ['handphone Samsung', 'handphone Xiaomi', 'iPhone',
                  'handphone Oppo', 'handphone Vivo', 'smartphone', 'HP android'],
    'Kunci': ['kunci motor', 'kunci rumah', 'kunci kos', 'gantungan kunci',
              'kunci loker', 'kunci sepeda', 'kunci gembok'],
    'Kartu Identitas': ['KTM', 'KTP', 'kartu perpustakaan', 'kartu akses gedung',
                        'SIM', 'kartu mahasiswa', 'ID card'],
    'Laptop': ['laptop Asus', 'laptop Lenovo', 'laptop HP', 'laptop Acer',
               'MacBook', 'charger laptop', 'laptop Dell'],
    'Aksesori': ['jam tangan', 'kacamata', 'gelang', 'cincin',
                 'kalung', 'topi', 'syal', 'ikat pinggang'],
    'Alat Tulis': ['pulpen', 'buku catatan', 'tempat pensil', 'kalkulator',
                   'penggaris', 'stabilo', 'pensil mekanik'],
    'Barang Lainnya': ['payung', 'botol minum', 'earphone', 'headset',
                       'flashdisk', 'power bank', 'kabel charger', 'mouse wireless']
}

# Variasi nama item untuk laporan temuan (agar tidak identik)
ITEM_ALIASES = {
    'dompet kulit': ['dompet berbahan kulit', 'dompet leather', 'dompet'],
    'dompet lipat': ['dompet model lipat', 'dompet bifold', 'dompet'],
    'tas ransel': ['ransel', 'backpack', 'tas punggung'],
    'tas selempang': ['sling bag', 'tas slempang', 'tas samping'],
    'tas laptop': ['tas untuk laptop', 'laptop bag', 'tas notebook'],
    'handphone Samsung': ['HP Samsung', 'Samsung', 'ponsel Samsung'],
    'handphone Xiaomi': ['HP Xiaomi', 'Xiaomi', 'ponsel Xiaomi'],
    'iPhone': ['HP Apple', 'handphone Apple', 'iPhone'],
    'handphone Oppo': ['HP Oppo', 'Oppo', 'ponsel Oppo'],
    'handphone Vivo': ['HP Vivo', 'Vivo', 'ponsel Vivo'],
    'kunci motor': ['kunci kendaraan', 'kunci motor matic', 'anak kunci motor'],
    'kunci rumah': ['kunci pintu', 'kunci', 'anak kunci'],
    'kunci kos': ['kunci kamar kos', 'kunci kamar', 'kunci pintu kos'],
    'gantungan kunci': ['keychain', 'key ring', 'gantungan'],
    'KTM': ['kartu tanda mahasiswa', 'kartu mahasiswa', 'KTM kampus'],
    'KTP': ['kartu identitas', 'kartu tanda penduduk', 'ID card'],
    'laptop Asus': ['Asus', 'notebook Asus', 'laptop merk Asus'],
    'laptop Lenovo': ['Lenovo', 'ThinkPad', 'laptop merk Lenovo'],
    'jam tangan': ['arloji', 'watch', 'jam'],
    'kacamata': ['glasses', 'kacamata baca', 'frame kacamata'],
    'botol minum': ['tumbler', 'botol air', 'bottle'],
    'earphone': ['earbuds', 'headset kecil', 'in-ear'],
    'power bank': ['portable charger', 'powerbank', 'PB'],
}

COLORS = [
    'hitam', 'putih', 'coklat', 'biru', 'merah', 'abu-abu',
    'hijau', 'kuning', 'pink', 'ungu', 'biru tua',
    'merah marun', 'krem', 'emas', 'perak'
]

COLOR_VARIATIONS = {
    'hitam': ['gelap', 'warna hitam', 'dark', 'hitam pekat'],
    'putih': ['warna terang', 'putih bersih', 'warna putih', 'putih susu'],
    'coklat': ['warna coklat', 'kecoklatan', 'brown', 'coklat tua'],
    'biru': ['warna biru', 'kebiruan', 'biru muda', 'biru langit'],
    'merah': ['warna merah', 'kemerahan', 'merah terang'],
    'abu-abu': ['warna abu', 'keabu-abuan', 'silver', 'abu muda'],
    'hijau': ['warna hijau', 'kehijauan', 'hijau tua', 'hijau muda'],
    'kuning': ['warna kuning', 'kekuningan', 'kuning cerah'],
    'pink': ['merah muda', 'warna pink', 'pink muda', 'rose'],
    'ungu': ['warna ungu', 'keunguan', 'ungu muda', 'violet'],
    'biru tua': ['navy', 'biru gelap', 'biru dongker', 'dark blue'],
    'merah marun': ['maroon', 'merah gelap', 'merah tua'],
    'krem': ['cream', 'broken white', 'krem muda', 'warna krem'],
    'emas': ['gold', 'keemasan', 'warna emas'],
    'perak': ['silver', 'keperakan', 'warna perak']
}

LOCATIONS = [
    'Perpustakaan Pusat', 'Kantin Fakultas Teknik', 'Gedung Kuliah Bersama',
    'Masjid Kampus', 'Laboratorium Komputer', 'Ruang Kelas A201',
    'Ruang Kelas B102', 'Taman Kampus', 'Parkiran Motor',
    'Parkiran Mobil', 'Fakultas Ekonomi', 'Fakultas Hukum',
    'Fakultas MIPA', 'Auditorium', 'Sekretariat BEM',
    'Kantin Pusat', 'Gedung Rektorat', 'Lapangan Olahraga',
    'Gazebo Kampus', 'Lorong Fakultas Teknik', 'Toilet Lantai 2 Gedung A',
    'Ruang Seminar', 'Koperasi Mahasiswa', 'Pos Satpam',
    'Halte Bus Kampus', 'Food Court', 'Lobby Gedung B',
    'Mushola Gedung C', 'Lab Fisika', 'Lab Kimia',
    'Studio Musik', 'UKM Center'
]

LOCATION_VARIATIONS = {
    'Perpustakaan Pusat': ['perpustakaan', 'perpus pusat', 'ruang baca perpustakaan', 'gedung perpustakaan'],
    'Kantin Fakultas Teknik': ['kantin teknik', 'kantin FT', 'depan kantin teknik', 'area kantin FT'],
    'Gedung Kuliah Bersama': ['GKB', 'gedung kuliah', 'ruang kuliah GKB', 'kelas di GKB'],
    'Masjid Kampus': ['masjid', 'tempat wudu masjid', 'area masjid', 'sekitar masjid kampus'],
    'Laboratorium Komputer': ['lab komputer', 'lab kom', 'ruang lab komputer', 'laboratorium IT'],
    'Kantin Pusat': ['kantin', 'area kantin', 'sekitar kantin pusat', 'depan kantin'],
    'Parkiran Motor': ['parkir motor', 'tempat parkir motor', 'area parkiran motor', 'parkiran'],
    'Parkiran Mobil': ['parkir mobil', 'tempat parkir', 'area parkir mobil'],
    'Taman Kampus': ['taman', 'area taman kampus', 'sekitar taman', 'dekat taman'],
    'Auditorium': ['aula', 'gedung auditorium', 'depan auditorium', 'area auditorium'],
    'Food Court': ['food court kampus', 'area food court', 'sekitar food court', 'dekat food court'],
    'Fakultas Ekonomi': ['FE', 'gedung ekonomi', 'area fakultas ekonomi'],
    'Fakultas Hukum': ['FH', 'gedung hukum', 'area fakultas hukum'],
    'Fakultas MIPA': ['FMIPA', 'gedung MIPA', 'area MIPA'],
    'Gazebo Kampus': ['gazebo', 'area gazebo', 'sekitar gazebo kampus'],
    'Lapangan Olahraga': ['lapangan', 'area olahraga', 'sekitar lapangan'],
    'Koperasi Mahasiswa': ['kopma', 'koperasi', 'toko kopma'],
    'Lobby Gedung B': ['lobby B', 'lobi gedung B', 'area lobby B'],
}

# Detail deskripsi: pasangan (lost_detail, found_detail)
# Didesain agar memiliki informasi serupa tapi kata-kata berbeda
DETAIL_PAIRS = {
    'Dompet': [
        ('berisi KTM dan kartu ATM BCA', 'ada kartu mahasiswa dan kartu bank di dalamnya'),
        ('model lipat dua, ada foto di dalamnya', 'bentuk lipat, terselip foto'),
        ('berisi uang dan beberapa kartu', 'ada uang tunai dan kartu-kartu'),
        ('ada resleting di bagian koin', 'terdapat saku koin dengan resleting'),
        ('merk Hush Puppies, ada inisial nama', 'bermerk, ada ukiran huruf di depan'),
        ('berisi SIM dan STNK motor', 'ada surat-surat kendaraan di dalamnya'),
        ('tipis, berisi kartu-kartu penting', 'dompet slim, isinya beberapa kartu'),
        ('baru dibeli, masih bagus kondisinya', 'kondisi masih baru dan terawat'),
        ('ada gantungan kecil di resleting', 'resletingnya ada hiasan kecil'),
        ('berisi kartu debit dan e-money', 'ada kartu bank dan kartu elektronik'),
    ],
    'Tas': [
        ('berisi buku kuliah dan laptop', 'ada buku dan laptop di dalamnya'),
        ('ada gantungan boneka kecil', 'terdapat gantungan karakter di tasnya'),
        ('berisi botol minum dan dompet', 'ada botol dan dompet di dalam'),
        ('merk Eiger, ada name tag', 'bermerk outdoor, ada label nama'),
        ('berisi alat tulis dan charger', 'isinya peralatan tulis dan kabel charger'),
        ('ada stiker di bagian depan', 'terdapat tempelan stiker di muka tas'),
        ('tali sudah agak lusuh', 'strap-nya sudah mulai pudar'),
        ('berisi jaket dan payung', 'ada jaket dan payung lipat di dalamnya'),
        ('ada saku tersembunyi di belakang', 'terdapat kantong rahasia di bagian punggung'),
        ('resleting utama agak macet', 'zipper utamanya sedikit bermasalah'),
    ],
    'Handphone': [
        ('dengan casing transparan', 'pakai case bening'),
        ('layar ada goresan kecil di pojok', 'layarnya sedikit lecet di sudut'),
        ('pakai screen protector', 'ada pelindung layar terpasang'),
        ('wallpaper foto kucing', 'background-nya gambar kucing'),
        ('ada stiker di belakang casing', 'di belakang case ada tempelan stiker'),
        ('memory 128GB, warna gradasi', 'penyimpanan besar, warna ombre'),
        ('masih ada sisa baterai saat hilang', 'HP masih dalam keadaan menyala'),
        ('ada pop socket di belakang', 'terpasang popsocket di bagian belakang'),
        ('casing motif marble', 'case bermotif marmer'),
        ('layar 6.5 inci, agak besar', 'ukuran layarnya lumayan lebar'),
    ],
    'Kunci': [
        ('gantungan kunci karakter doraemon', 'ada gantungan boneka biru di kuncinya'),
        ('ada 3 anak kunci', 'terdiri dari tiga buah kunci'),
        ('kunci motor Honda Beat', 'kunci kendaraan matic Honda'),
        ('gantungan kunci dari kulit', 'key chain berbahan leather'),
        ('ada karet warna-warni di gagangnya', 'gagang kunci diberi tanda warna'),
        ('kunci duplikat, bentuknya sederhana', 'kunci salinan, model simpel'),
        ('gantungan kunci berbentuk hati', 'keychain berbentuk love'),
        ('ada label nomor kamar', 'tertulis nomor di gantungannya'),
        ('kunci motor Yamaha NMAX', 'kunci motor matic Yamaha'),
        ('gantungan kunci souvenir kampus', 'ada gantungan merchandise universitas'),
    ],
    'Kartu Identitas': [
        ('atas nama mahasiswa teknik informatika', 'milik mahasiswa jurusan IT'),
        ('KTM semester genap 2026', 'kartu mahasiswa periode genap tahun ini'),
        ('ada foto dan NIM tertera jelas', 'tercantum foto dan nomor induk mahasiswa'),
        ('kartu masih baru, belum pudar', 'kondisi kartu masih bagus'),
        ('kartu perpustakaan dan KTM sekaligus', 'ada dua kartu sekaligus, perpus dan mahasiswa'),
        ('SIM C motor, baru perpanjang', 'surat izin mengemudi roda dua, masih berlaku'),
        ('kartu akses gedung laboratorium', 'card khusus untuk masuk lab'),
        ('ada gantungan tali di kartu', 'kartu dengan lanyard terpasang'),
        ('KTP asli, bukan fotokopi', 'kartu identitas original'),
        ('kartu dengan foto lama', 'di kartu terdapat foto yang sudah agak lama'),
    ],
    'Laptop': [
        ('ada stiker di bagian cover', 'terdapat stiker-stiker di penutupnya'),
        ('charger original bawaan laptop', 'adaptor asli masih terbawa'),
        ('RAM 8GB, SSD 512GB', 'spesifikasi menengah, performa cukup baik'),
        ('dalam tas laptop hitam', 'di dalam soft case berwarna gelap'),
        ('layar 14 inci, desain tipis', 'ukuran layar sedang, model slim'),
        ('ada goresan kecil di sudut bodi', 'sedikit lecet di bagian pojok casing'),
        ('masih ada data tugas kuliah di dalamnya', 'isinya file-file tugas dan dokumen akademik'),
        ('baterai sudah mulai drop', 'daya tahan baterainya agak menurun'),
        ('keyboard ada tombol yang agak lengket', 'beberapa tombol keyboard sedikit bermasalah'),
        ('ada engsel yang sedikit longgar', 'hinge-nya agak kendur'),
    ],
    'Aksesori': [
        ('jam tangan digital merk Casio', 'arloji digital bermerk Casio'),
        ('kacamata minus, frame tipis', 'glasses untuk rabun jauh, bingkai slim'),
        ('gelang tali warna-warni', 'bracelet dari benang berwarna-warni'),
        ('cincin polos tanpa ukiran', 'ring sederhana tanpa motif'),
        ('topi baseball dengan logo', 'cap dengan emblem di depan'),
        ('kacamata hitam model cat eye', 'sunglasses bentuk mata kucing'),
        ('jam tangan tali kulit', 'arloji dengan strap leather'),
        ('kalung liontin kecil', 'necklace dengan pendant mini'),
        ('gelang rantai tipis', 'bracelet chain model halus'),
        ('topi bucket hat polos', 'bucket hat tanpa motif'),
    ],
    'Alat Tulis': [
        ('pulpen parker warna silver', 'pen bermerk Parker, warna metalik'),
        ('buku catatan spiral A5', 'notebook spiral ukuran sedang'),
        ('tempat pensil motif bunga', 'pencil case bermotif floral'),
        ('kalkulator scientific Casio', 'kalkulator ilmiah bermerk Casio'),
        ('set stabilo 6 warna', 'highlighter set berisi enam warna'),
        ('pensil mekanik 0.5mm', 'mechanical pencil ukuran halus'),
        ('buku catatan kulit coklat', 'notebook dengan cover leather'),
        ('penggaris 30cm transparan', 'ruler bening ukuran panjang'),
        ('pulpen multiwarna 4 in 1', 'pen yang bisa ganti warna'),
        ('buku agenda harian', 'planner daily notebook'),
    ],
    'Barang Lainnya': [
        ('payung lipat otomatis', 'payung yang bisa dilipat, auto open'),
        ('botol minum tumbler stainless', 'tumbler dari bahan stainless steel'),
        ('earphone wireless bluetooth', 'earbuds tanpa kabel, koneksi bluetooth'),
        ('flashdisk 32GB merk Sandisk', 'USB drive 32 giga bermerk Sandisk'),
        ('power bank 10000mAh', 'portable charger kapasitas sepuluh ribu'),
        ('kabel charger type C', 'kabel USB-C untuk pengisian daya'),
        ('mouse wireless Logitech', 'mouse tanpa kabel bermerk Logitech'),
        ('headset gaming dengan mic', 'headphone gaming ada mikrofonnya'),
        ('payung besar warna polos', 'payung ukuran besar tanpa motif'),
        ('flash disk kecil model slide', 'USB flash drive model geser'),
    ],
}

# ============================================================
# TEMPLATE DESKRIPSI
# ============================================================

LOST_TEMPLATES = [
    "Saya kehilangan {item} {color} di {location}. {detail}",
    "Tolong bantu, {item} saya yang {color} tertinggal di {location}. {detail}",
    "{item} {color} saya hilang di sekitar {location}. {detail}",
    "Siapa yang menemukan {item} {color} di {location}? {detail}",
    "Mohon bantuan, saya kehilangan {item} berwarna {color} di {location}. {detail}",
    "Saya tidak sengaja meninggalkan {item} {color} di {location}. {detail}",
    "Kehilangan {item} warna {color} sekitar area {location}. {detail}",
    "Butuh bantuan! {item} {color} milik saya hilang di {location}. {detail}",
    "Apakah ada yang menemukan {item} {color} di {location}? {detail}. Mohon hubungi saya",
    "Baru saja kehilangan {item} berwarna {color} saat berada di {location}. {detail}",
    "Saya lupa menaruh {item} {color} di {location}. {detail}. Sangat butuh",
    "{item} warna {color} milik saya tertinggal di {location}. {detail}",
]

FOUND_TEMPLATES = [
    "Menemukan {item} {color} di {location}. {detail}",
    "Ditemukan {item} {color} di area {location}. {detail}",
    "Ada yang kehilangan {item} {color}? Ditemukan di {location}. {detail}",
    "Barang ditemukan: {item} {color} di {location}. {detail}",
    "Saya menemukan {item} berwarna {color} di {location}. {detail}",
    "Ditemukan {item} warna {color} tertinggal di {location}. {detail}",
    "Ada {item} {color} tertinggal di {location}. {detail}. Silakan hubungi untuk klaim",
    "Tadi menemukan {item} {color} di sekitar {location}. {detail}",
    "Barang temuan: {item} berwarna {color} di {location}. {detail}",
    "Ditemukan sebuah {item} {color} yang tertinggal di {location}. {detail}",
    "Siapa pemilik {item} {color} ini? Ditemukan di {location}. {detail}",
    "Ketemu {item} {color} di {location}. {detail}. Silakan klaim",
]

UNMATCHED_EXTRA_LOST = [
    "Kehilangan {item} {color} di {location} pada tanggal {date_str}. {detail}. Sangat penting bagi saya",
    "Tolong, {item} saya {color} hilang di {location}. {detail}. Terakhir terlihat tanggal {date_str}",
    "{item} {color} milik saya tidak ada saat kembali ke {location}. {detail}",
    "Saya yakin menaruh {item} {color} di {location} tapi sekarang sudah tidak ada. {detail}",
    "Mohon bantuannya, {item} berwarna {color} tertinggal di {location}. {detail}. Sangat butuh",
    "Saya baru sadar {item} {color} saya tidak ada. Terakhir di {location}. {detail}",
    "Hilang: {item} {color} dari {location}. {detail}. Jika menemukan tolong hubungi",
    "{item} {color} saya jatuh atau tertinggal di {location}. {detail}",
    "Tolong bantu cari {item} {color} yang hilang di {location}. {detail}",
    "Lapor kehilangan {item} {color} di {location}, tanggal {date_str}. {detail}",
]

UNMATCHED_EXTRA_FOUND = [
    "Menemukan {item} {color} yang ditinggalkan di {location}. {detail}",
    "Ada {item} {color} tak bertuan di {location}. {detail}. Silakan klaim",
    "Barang temuan di {location}: {item} {color}. {detail}",
    "Tadi pagi menemukan {item} berwarna {color} di {location}. {detail}",
    "Ditemukan {item} {color} dekat {location}. {detail}. Hubungi untuk klaim",
    "Siapa punya {item} {color}? Ketemu di {location}. {detail}",
    "Saya mau melaporkan penemuan {item} {color} di {location}. {detail}",
    "Ada yang merasa kehilangan {item} {color} di sekitar {location}? {detail}",
    "Lapor penemuan: {item} {color} di {location}. {detail}",
    "Ketemu {item} {color} di {location} tadi sore. {detail}",
]

# ============================================================
# UTILITAS
# ============================================================

BULAN_ID = {
    1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April',
    5: 'Mei', 6: 'Juni', 7: 'Juli', 8: 'Agustus',
    9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'
}

def format_tanggal(dt):
    """Format tanggal dalam Bahasa Indonesia."""
    return f"{dt.day} {BULAN_ID[dt.month]} {dt.year}"

def get_random_date():
    """Generate tanggal acak dalam 3 bulan terakhir."""
    start = datetime(2026, 7, 1)
    end = datetime(2026, 10, 9)
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))

def get_color_variation(color):
    """Dapatkan variasi/sinonim warna."""
    if color in COLOR_VARIATIONS:
        return random.choice(COLOR_VARIATIONS[color])
    return color

def get_location_variation(location):
    """Dapatkan variasi/sinonim lokasi."""
    if location in LOCATION_VARIATIONS:
        return random.choice(LOCATION_VARIATIONS[location])
    return location

def get_item_alias(item):
    """Dapatkan alias/variasi nama item."""
    if item in ITEM_ALIASES:
        return random.choice(ITEM_ALIASES[item])
    return item

# ============================================================
# GENERATOR UTAMA
# ============================================================

def generate_matched_pairs(num_groups):
    """Generate pasangan laporan hilang-temuan yang cocok."""
    reports = []

    for group_id in range(1, num_groups + 1):
        # Distribusi kategori merata
        category = CATEGORIES[(group_id - 1) % len(CATEGORIES)]
        item = random.choice(ITEMS_PER_CATEGORY[category])
        color = random.choice(COLORS)
        location = random.choice(LOCATIONS)
        base_date = get_random_date()

        # Pilih pasangan detail
        detail_idx = random.randint(0, len(DETAIL_PAIRS[category]) - 1)
        lost_detail, found_detail = DETAIL_PAIRS[category][detail_idx]

        # --- Laporan HILANG ---
        lost_template = random.choice(LOST_TEMPLATES)
        lost_desc = lost_template.format(
            item=item, color=color, location=location, detail=lost_detail
        )
        reports.append({
            'report_type': 'lost',
            'item_name': item,
            'category': category,
            'description': lost_desc,
            'color': color,
            'location': location,
            'event_date': base_date.strftime('%Y-%m-%d'),
            'match_group': group_id
        })

        # --- Laporan TEMUAN (dengan variasi) ---
        color_var = get_color_variation(color)
        loc_var = get_location_variation(location)
        item_var = get_item_alias(item)
        found_date = base_date + timedelta(days=random.randint(0, 2))

        found_template = random.choice(FOUND_TEMPLATES)
        found_desc = found_template.format(
            item=item_var, color=color_var, location=loc_var, detail=found_detail
        )
        reports.append({
            'report_type': 'found',
            'item_name': item_var,
            'category': category,
            'description': found_desc,
            'color': color_var,
            'location': loc_var,
            'event_date': found_date.strftime('%Y-%m-%d'),
            'match_group': group_id
        })

    return reports


def generate_unmatched_reports(num_reports):
    """Generate laporan tanpa pasangan."""
    reports = []

    for i in range(num_reports):
        category = CATEGORIES[i % len(CATEGORIES)]
        item = random.choice(ITEMS_PER_CATEGORY[category])
        color = random.choice(COLORS)
        location = random.choice(LOCATIONS)
        date = get_random_date()
        report_type = 'lost' if i % 2 == 0 else 'found'

        # Pilih detail acak (satu sisi saja)
        detail_pair = random.choice(DETAIL_PAIRS[category])
        detail = detail_pair[0] if report_type == 'lost' else detail_pair[1]
        date_str = format_tanggal(date)

        if report_type == 'lost':
            template = random.choice(UNMATCHED_EXTRA_LOST)
        else:
            template = random.choice(UNMATCHED_EXTRA_FOUND)

        desc = template.format(
            item=item, color=color, location=location,
            detail=detail, date_str=date_str
        )

        reports.append({
            'report_type': report_type,
            'item_name': item,
            'category': category,
            'description': desc,
            'color': color,
            'location': location,
            'event_date': date.strftime('%Y-%m-%d'),
            'match_group': 0
        })

    return reports


def main():
    print("=" * 60)
    print("LostLink Dataset Generator")
    print("=" * 60)

    # Generate semua laporan
    matched = generate_matched_pairs(NUM_MATCH_GROUPS)
    unmatched = generate_unmatched_reports(NUM_UNMATCHED)
    all_reports = matched + unmatched

    # Acak urutan
    random.shuffle(all_reports)

    # Assign report_id
    for i, report in enumerate(all_reports, 1):
        report['report_id'] = f'RPT-{i:04d}'

    # Tulis ke CSV
    fieldnames = [
        'report_id', 'report_type', 'item_name', 'category',
        'description', 'color', 'location', 'event_date', 'match_group'
    ]

    output_path = 'lostlink_dataset.csv'
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_reports)

    # Statistik
    lost_count = sum(1 for r in all_reports if r['report_type'] == 'lost')
    found_count = sum(1 for r in all_reports if r['report_type'] == 'found')
    matched_count = sum(1 for r in all_reports if r['match_group'] > 0)
    unique_groups = len(set(r['match_group'] for r in all_reports if r['match_group'] > 0))

    print(f"\n[OK] Dataset berhasil dibuat: {output_path}")
    print(f"  Total laporan  : {len(all_reports)}")
    print(f"  Barang hilang  : {lost_count}")
    print(f"  Barang temuan  : {found_count}")
    print(f"  Berpasangan    : {matched_count} ({unique_groups} grup)")
    print(f"  Tanpa pasangan : {len(all_reports) - matched_count}")
    print(f"  Jumlah kategori: {len(set(r['category'] for r in all_reports))}")

    # Distribusi kategori
    cat_counts = Counter(r['category'] for r in all_reports)
    print("\nDistribusi kategori:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        bar = '#' * (count // 3)
        print(f"  {cat:20s}: {count:3d} {bar}")

    # Contoh data
    print("\nContoh 3 laporan pertama:")
    for r in all_reports[:3]:
        print(f"  [{r['report_id']}] {r['report_type'].upper():5s} | {r['category']:15s} | {r['description'][:80]}...")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
