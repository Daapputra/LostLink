#!/usr/bin/env python3
"""
LostLink Model Training Script
===============================
Melatih dan mengevaluasi dua model NLP:
1. TF-IDF + Logistic Regression (klasifikasi kategori)
2. Word Embedding + LSTM (klasifikasi kategori)

Hasil training disimpan di folder trained_models/ untuk digunakan oleh backend.

Cara menjalankan:
    cd backend
    python train_models.py
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.nlp.preprocessing import (
    preprocess_for_tfidf, preprocess_for_embedding,
    preprocess_dataframe, show_preprocessing_examples
)

# scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import joblib

# ============================================================
# KONFIGURASI
# ============================================================

DATASET_PATH = os.path.join('..', 'dataset', 'lostlink_dataset.csv')
MODEL_DIR = os.path.join('trained_models')
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.15  # dari training set, untuk LSTM

# TF-IDF config
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)

# LSTM config
LSTM_MAX_WORDS = 5000
LSTM_MAX_LEN = 100
LSTM_EMBEDDING_DIM = 64
LSTM_LSTM_UNITS = 64
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32


def ensure_dirs():
    """Buat folder yang diperlukan."""
    os.makedirs(MODEL_DIR, exist_ok=True)


def load_dataset():
    """Muat dan periksa dataset."""
    print("\n" + "=" * 60)
    print("1. MEMUAT DATASET")
    print("=" * 60)

    df = pd.read_csv(DATASET_PATH, encoding='utf-8')
    print(f"   Dataset dimuat: {len(df)} baris")
    print(f"   Kolom: {list(df.columns)}")
    print(f"   Tipe laporan: {df['report_type'].value_counts().to_dict()}")
    print(f"   Kategori: {df['category'].nunique()} jenis")
    print(f"   Missing values:\n{df.isnull().sum().to_string()}")

    return df


def preprocess_dataset(df):
    """Preprocess dataset untuk kedua model."""
    print("\n" + "=" * 60)
    print("2. PREPROCESSING DATA")
    print("=" * 60)

    # Tampilkan contoh sebelum dan sesudah
    sample_texts = df['description'].head(5).tolist()
    show_preprocessing_examples(sample_texts[:3], method='tfidf')

    # Preprocess untuk TF-IDF
    df_tfidf = preprocess_dataframe(df, text_column='description', method='tfidf')
    print(f"   TF-IDF preprocessing: {len(df_tfidf)} baris")

    # Preprocess untuk Embedding
    df_emb = preprocess_dataframe(df, text_column='description', method='embedding')
    print(f"   Embedding preprocessing: {len(df_emb)} baris")

    return df_tfidf, df_emb


def train_tfidf_model(df):
    """
    Eksperimen 1: TF-IDF + Logistic Regression
    Tugas: Klasifikasi kategori barang berdasarkan deskripsi.
    """
    print("\n" + "=" * 60)
    print("3. EKSPERIMEN 1: TF-IDF + LOGISTIC REGRESSION")
    print("=" * 60)

    results = {}
    start_time = time.time()

    # Siapkan data
    X_text = df['description_clean'].values
    y = df['category'].values

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    categories = label_encoder.classes_.tolist()
    print(f"   Kategori ({len(categories)}): {categories}")

    # Split data (PENTING: preprocessing hanya dipelajari dari data training)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y_encoded, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y_encoded
    )
    print(f"   Training: {len(X_train_text)} | Testing: {len(X_test_text)}")

    # TF-IDF Vectorization (fit hanya pada training data!)
    print("   Membuat fitur TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=True
    )
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
    X_test_tfidf = tfidf_vectorizer.transform(X_test_text)
    print(f"   Dimensi fitur TF-IDF: {X_train_tfidf.shape[1]}")

    # Latih Logistic Regression
    print("   Melatih Logistic Regression...")
    train_start = time.time()
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver='lbfgs',
        C=1.0
    )
    lr_model.fit(X_train_tfidf, y_train)
    train_time = time.time() - train_start
    print(f"   Waktu pelatihan: {train_time:.2f} detik")

    # Prediksi
    infer_start = time.time()
    y_pred = lr_model.predict(X_test_tfidf)
    infer_time = time.time() - infer_start
    print(f"   Waktu inferensi: {infer_time:.4f} detik")

    # Evaluasi
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"\n   === HASIL EVALUASI ===")
    print(f"   Accuracy : {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall   : {recall:.4f}")
    print(f"   F1-Score : {f1:.4f}")

    # Classification Report
    report = classification_report(
        y_test, y_pred,
        target_names=categories,
        output_dict=True,
        zero_division=0
    )
    print(f"\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=categories, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"   Confusion Matrix:")
    print(f"   {cm}")

    # Simpan model dan artefak
    print("\n   Menyimpan model...")
    joblib.dump(lr_model, os.path.join(MODEL_DIR, 'tfidf_lr_model.joblib'))
    joblib.dump(tfidf_vectorizer, os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib'))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, 'label_encoder.joblib'))

    total_time = time.time() - start_time

    # Simpan hasil evaluasi
    results = {
        'model': 'TF-IDF + Logistic Regression',
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'train_time_seconds': round(train_time, 4),
        'inference_time_seconds': round(infer_time, 4),
        'total_time_seconds': round(total_time, 4),
        'train_size': len(X_train_text),
        'test_size': len(X_test_text),
        'num_features': X_train_tfidf.shape[1],
        'categories': categories,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'timestamp': datetime.now().isoformat()
    }

    with open(os.path.join(MODEL_DIR, 'tfidf_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"   [OK] Model TF-IDF + LR berhasil disimpan!")
    return results


def train_lstm_model(df):
    """
    Eksperimen 2: Word Embedding + LSTM
    Tugas: Klasifikasi kategori barang berdasarkan deskripsi.
    """
    print("\n" + "=" * 60)
    print("4. EKSPERIMEN 2: WORD EMBEDDING + LSTM")
    print("=" * 60)

    results = {}
    start_time = time.time()

    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.utils import to_categorical

        print(f"   TensorFlow version: {tf.__version__}")
    except ImportError as e:
        print(f"   [ERROR] TensorFlow tidak tersedia: {e}")
        print(f"   Melewati eksperimen LSTM.")
        results = {
            'model': 'Word Embedding + LSTM',
            'status': 'skipped',
            'reason': f'TensorFlow not available: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }
        with open(os.path.join(MODEL_DIR, 'lstm_results.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return results

    # Siapkan data
    X_text = df['description_clean'].values
    y = df['category'].values

    # Encode labels
    label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.joblib'))
    y_encoded = label_encoder.transform(y)
    num_classes = len(label_encoder.classes_)
    categories = label_encoder.classes_.tolist()

    # Split data training/testing (sama dengan TF-IDF untuk perbandingan fair)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y_encoded, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y_encoded
    )

    # Split training menjadi train/validation
    X_train_text, X_val_text, y_train, y_val = train_test_split(
        X_train_text, y_train, test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE, stratify=y_train
    )

    print(f"   Training  : {len(X_train_text)}")
    print(f"   Validation: {len(X_val_text)}")
    print(f"   Testing   : {len(X_test_text)}")

    # Tokenisasi (fit hanya pada data training!)
    print("   Melakukan tokenisasi...")
    tokenizer = Tokenizer(num_words=LSTM_MAX_WORDS, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train_text)

    X_train_seq = tokenizer.texts_to_sequences(X_train_text)
    X_val_seq = tokenizer.texts_to_sequences(X_val_text)
    X_test_seq = tokenizer.texts_to_sequences(X_test_text)

    # Padding
    X_train_pad = pad_sequences(X_train_seq, maxlen=LSTM_MAX_LEN, padding='post', truncating='post')
    X_val_pad = pad_sequences(X_val_seq, maxlen=LSTM_MAX_LEN, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=LSTM_MAX_LEN, padding='post', truncating='post')

    vocab_size = min(LSTM_MAX_WORDS, len(tokenizer.word_index) + 1)
    print(f"   Vocabulary size: {vocab_size}")
    print(f"   Sequence length: {LSTM_MAX_LEN}")

    # One-hot encode labels
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_val_cat = to_categorical(y_val, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)

    # Bangun model LSTM
    print("   Membangun arsitektur LSTM...")
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=LSTM_EMBEDDING_DIM, input_length=LSTM_MAX_LEN),
        Bidirectional(LSTM(LSTM_LSTM_UNITS, return_sequences=False)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Early Stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    # Latih model
    print("\n   Melatih model LSTM...")
    train_start = time.time()
    history = model.fit(
        X_train_pad, y_train_cat,
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        validation_data=(X_val_pad, y_val_cat),
        callbacks=[early_stop],
        verbose=1
    )
    train_time = time.time() - train_start
    print(f"   Waktu pelatihan: {train_time:.2f} detik")
    print(f"   Epoch terbaik: {len(history.history['loss'])}")

    # Evaluasi
    print("\n   Evaluasi pada data testing...")
    infer_start = time.time()
    y_pred_proba = model.predict(X_test_pad, verbose=0)
    infer_time = time.time() - infer_start
    y_pred = np.argmax(y_pred_proba, axis=1)
    print(f"   Waktu inferensi: {infer_time:.4f} detik")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"\n   === HASIL EVALUASI ===")
    print(f"   Accuracy : {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall   : {recall:.4f}")
    print(f"   F1-Score : {f1:.4f}")

    # Classification Report
    report = classification_report(
        y_test, y_pred,
        target_names=categories,
        output_dict=True,
        zero_division=0
    )
    print(f"\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=categories, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"   Confusion Matrix:")
    print(f"   {cm}")

    # Simpan model
    print("\n   Menyimpan model...")
    model.save(os.path.join(MODEL_DIR, 'lstm_model.keras'))

    # Simpan tokenizer config
    tokenizer_config = {
        'word_index': dict(list(tokenizer.word_index.items())[:LSTM_MAX_WORDS]),
        'max_len': LSTM_MAX_LEN,
        'max_words': LSTM_MAX_WORDS,
        'oov_token': '<OOV>'
    }
    with open(os.path.join(MODEL_DIR, 'tokenizer_config.json'), 'w', encoding='utf-8') as f:
        json.dump(tokenizer_config, f, ensure_ascii=False)

    # Simpan tokenizer sebagai pickle juga
    import pickle
    with open(os.path.join(MODEL_DIR, 'tokenizer.pickle'), 'wb') as f:
        pickle.dump(tokenizer, f)

    total_time = time.time() - start_time

    # Simpan training history
    history_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}

    # Simpan hasil evaluasi
    results = {
        'model': 'Word Embedding + LSTM',
        'status': 'completed',
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'train_time_seconds': round(train_time, 4),
        'inference_time_seconds': round(infer_time, 4),
        'total_time_seconds': round(total_time, 4),
        'train_size': len(X_train_text),
        'validation_size': len(X_val_text),
        'test_size': len(X_test_text),
        'vocab_size': vocab_size,
        'embedding_dim': LSTM_EMBEDDING_DIM,
        'lstm_units': LSTM_LSTM_UNITS,
        'max_sequence_length': LSTM_MAX_LEN,
        'epochs_trained': len(history.history['loss']),
        'categories': categories,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'training_history': history_data,
        'timestamp': datetime.now().isoformat()
    }

    with open(os.path.join(MODEL_DIR, 'lstm_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"   [OK] Model LSTM berhasil disimpan!")
    return results


def compare_models(tfidf_results, lstm_results):
    """Bandingkan hasil kedua model."""
    print("\n" + "=" * 60)
    print("5. PERBANDINGAN MODEL")
    print("=" * 60)

    print(f"\n   {'Metrik':<25} {'TF-IDF + LR':<15} {'Emb + LSTM':<15}")
    print(f"   {'-'*55}")

    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        tfidf_val = tfidf_results.get(metric, 'N/A')
        lstm_val = lstm_results.get(metric, 'N/A')

        if isinstance(tfidf_val, float) and isinstance(lstm_val, float):
            better = '<--' if tfidf_val >= lstm_val else '-->'
            print(f"   {metric:<25} {tfidf_val:<15.4f} {lstm_val:<15.4f} {better}")
        else:
            print(f"   {metric:<25} {str(tfidf_val):<15} {str(lstm_val):<15}")

    for metric in ['train_time_seconds', 'inference_time_seconds']:
        tfidf_val = tfidf_results.get(metric, 'N/A')
        lstm_val = lstm_results.get(metric, 'N/A')
        if isinstance(tfidf_val, float) and isinstance(lstm_val, float):
            print(f"   {metric:<25} {tfidf_val:<15.4f} {lstm_val:<15.4f}")
        else:
            print(f"   {metric:<25} {str(tfidf_val):<15} {str(lstm_val):<15}")

    # Simpan perbandingan
    comparison = {
        'tfidf_lr': tfidf_results,
        'lstm': lstm_results,
        'timestamp': datetime.now().isoformat(),
        'notes': [
            'Kedua model dievaluasi pada split data yang sama (test_size=0.2, random_state=42).',
            'Dataset sintetis dengan distribusi kategori yang relatif seimbang.',
            'TF-IDF menggunakan unigram dan bigram.',
            'LSTM menggunakan Bidirectional LSTM dengan embedding layer yang dilatih dari dataset.',
            'Perbandingan ini bersifat eksperimental dan tidak dapat digeneralisasi ke dataset lain.',
        ]
    }

    with open(os.path.join(MODEL_DIR, 'comparison_results.json'), 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print(f"\n   [OK] Hasil perbandingan disimpan!")


def main():
    print("=" * 60)
    print("LostLink Model Training")
    print(f"Waktu mulai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    ensure_dirs()

    # 1. Muat dataset
    df = load_dataset()

    # 2. Preprocessing
    df_tfidf, df_emb = preprocess_dataset(df)

    # 3. TF-IDF + Logistic Regression
    tfidf_results = train_tfidf_model(df_tfidf)

    # 4. Word Embedding + LSTM
    lstm_results = train_lstm_model(df_emb)

    # 5. Perbandingan
    compare_models(tfidf_results, lstm_results)

    print("\n" + "=" * 60)
    print(f"TRAINING SELESAI")
    print(f"Waktu selesai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model tersimpan di: {os.path.abspath(MODEL_DIR)}")
    print("=" * 60)


if __name__ == '__main__':
    main()
