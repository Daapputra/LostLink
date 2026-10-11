"""
LostLink TF-IDF Model Module
=============================
Modul untuk loading dan penggunaan model TF-IDF + Logistic Regression
dan cosine similarity untuk matching laporan.
"""

import os
import json
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from .preprocessing import preprocess_for_tfidf, combine_text_features

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'trained_models')


class TFIDFModel:
    """Wrapper untuk model TF-IDF + Logistic Regression."""

    def __init__(self):
        self.vectorizer = None
        self.classifier = None
        self.label_encoder = None
        self.results = None
        self.is_loaded = False

    def load(self):
        """Muat model dari disk."""
        try:
            vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')
            model_path = os.path.join(MODEL_DIR, 'tfidf_lr_model.joblib')
            encoder_path = os.path.join(MODEL_DIR, 'label_encoder.joblib')

            if not all(os.path.exists(p) for p in [vectorizer_path, model_path, encoder_path]):
                print("[TF-IDF] Model files not found.")
                return False

            self.vectorizer = joblib.load(vectorizer_path)
            self.classifier = joblib.load(model_path)
            self.label_encoder = joblib.load(encoder_path)

            # Load evaluation results
            results_path = os.path.join(MODEL_DIR, 'tfidf_results.json')
            if os.path.exists(results_path):
                with open(results_path, 'r', encoding='utf-8') as f:
                    self.results = json.load(f)

            self.is_loaded = True
            print("[TF-IDF] Model loaded successfully.")
            return True

        except Exception as e:
            print(f"[TF-IDF] Error loading model: {e}")
            return False

    def predict_category(self, text: str) -> dict:
        """
        Prediksi kategori barang dari deskripsi teks.

        Returns:
            dict: {'category': str, 'confidence': float, 'probabilities': dict}
        """
        if not self.is_loaded:
            return {'error': 'Model not loaded'}

        clean_text = preprocess_for_tfidf(text)
        tfidf_vector = self.vectorizer.transform([clean_text])

        # Prediksi
        prediction = self.classifier.predict(tfidf_vector)[0]
        probabilities = self.classifier.predict_proba(tfidf_vector)[0]

        category = self.label_encoder.inverse_transform([prediction])[0]
        confidence = float(max(probabilities))

        # Semua probabilitas
        all_probs = {}
        for i, cat in enumerate(self.label_encoder.classes_):
            all_probs[cat] = round(float(probabilities[i]), 4)

        return {
            'category': category,
            'confidence': round(confidence, 4),
            'probabilities': all_probs
        }

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Hitung cosine similarity antara dua teks menggunakan TF-IDF."""
        if not self.is_loaded:
            return 0.0

        clean1 = preprocess_for_tfidf(text1)
        clean2 = preprocess_for_tfidf(text2)

        vectors = self.vectorizer.transform([clean1, clean2])
        sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return float(sim)

    def find_matches(self, source_report: dict, candidate_reports: list,
                     threshold: float = 0.1, top_k: int = 10) -> list:
        """
        Cari kandidat kecocokan untuk laporan sumber.

        Args:
            source_report: Laporan yang dicari kecocokannya.
            candidate_reports: Daftar laporan kandidat.
            threshold: Ambang batas skor minimum.
            top_k: Jumlah kandidat teratas.

        Returns:
            list: Daftar kandidat dengan skor kecocokan.
        """
        if not self.is_loaded or not candidate_reports:
            return []

        # Gabungkan fitur teks dari source
        source_text = self._combine_report_text(source_report)
        source_clean = preprocess_for_tfidf(source_text)

        # Gabungkan fitur teks dari semua kandidat
        candidate_texts = [preprocess_for_tfidf(self._combine_report_text(r)) for r in candidate_reports]

        # Hitung TF-IDF vectors
        all_texts = [source_clean] + candidate_texts
        tfidf_matrix = self.vectorizer.transform(all_texts)

        # Cosine similarity
        source_vector = tfidf_matrix[0:1]
        candidate_vectors = tfidf_matrix[1:]
        similarities = cosine_similarity(source_vector, candidate_vectors)[0]

        # Buat hasil dengan scoring tambahan
        results = []
        for i, (report, tfidf_score) in enumerate(zip(candidate_reports, similarities)):
            if tfidf_score < threshold:
                continue

            # Skor tambahan
            category_match = source_report.get('category', '') == report.get('category', '')
            color_match = self._color_similarity(source_report.get('color', ''), report.get('color', ''))
            location_sim = self._location_similarity(source_report.get('location', ''), report.get('location', ''))

            # Skor gabungan (weighted)
            combined = (
                tfidf_score * 0.5 +
                (1.0 if category_match else 0.0) * 0.2 +
                color_match * 0.15 +
                location_sim * 0.15
            )

            # Penjelasan
            explanation = self._generate_explanation(
                source_report, report, tfidf_score, category_match, color_match, location_sim
            )

            results.append({
                'report': report,
                'tfidf_score': round(float(tfidf_score), 4),
                'combined_score': round(float(combined), 4),
                'category_match': category_match,
                'color_match': color_match > 0.5,
                'location_similarity': round(float(location_sim), 4),
                'explanation': explanation
            })

        # Sort by combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        return results[:top_k]

    def _combine_report_text(self, report: dict) -> str:
        """Gabungkan semua field teks dari laporan."""
        parts = [
            report.get('description', ''),
            report.get('item_name', ''),
            report.get('category', ''),
            report.get('color', ''),
            report.get('location', '')
        ]
        return ' '.join(p for p in parts if p)

    def _color_similarity(self, color1: str, color2: str) -> float:
        """Hitung kesamaan warna sederhana."""
        if not color1 or not color2:
            return 0.0
        c1 = color1.lower().strip()
        c2 = color2.lower().strip()
        if c1 == c2:
            return 1.0

        # Sinonim warna
        color_groups = [
            {'hitam', 'gelap', 'dark', 'hitam pekat'},
            {'putih', 'terang', 'putih bersih', 'putih susu'},
            {'coklat', 'kecoklatan', 'brown', 'coklat tua', 'coklat muda'},
            {'biru', 'kebiruan', 'biru muda', 'biru langit'},
            {'merah', 'kemerahan', 'merah terang'},
            {'abu-abu', 'abu', 'silver', 'keabu-abuan'},
            {'biru tua', 'navy', 'biru gelap', 'biru dongker', 'dark blue'},
            {'merah marun', 'maroon', 'merah gelap', 'merah tua'},
            {'pink', 'merah muda', 'rose'},
        ]

        for group in color_groups:
            if c1 in group and c2 in group:
                return 0.8
            # Partial match
            for word in group:
                if c1 in word or word in c1:
                    for word2 in group:
                        if c2 in word2 or word2 in c2:
                            return 0.6

        return 0.0

    def _location_similarity(self, loc1: str, loc2: str) -> float:
        """Hitung kesamaan lokasi sederhana."""
        if not loc1 or not loc2:
            return 0.0
        l1 = loc1.lower().strip()
        l2 = loc2.lower().strip()
        if l1 == l2:
            return 1.0

        # Word overlap
        words1 = set(l1.split())
        words2 = set(l2.split())
        if not words1 or not words2:
            return 0.0

        overlap = len(words1 & words2)
        total = len(words1 | words2)
        return overlap / total if total > 0 else 0.0

    def _generate_explanation(self, source, candidate, tfidf_score,
                              category_match, color_match, location_sim):
        """Generate penjelasan mengapa kandidat direkomendasikan."""
        reasons = []

        if tfidf_score > 0.3:
            reasons.append(f"Deskripsi sangat mirip (skor: {tfidf_score:.2f})")
        elif tfidf_score > 0.15:
            reasons.append(f"Deskripsi cukup mirip (skor: {tfidf_score:.2f})")
        else:
            reasons.append(f"Ada kesamaan deskripsi (skor: {tfidf_score:.2f})")

        if category_match:
            reasons.append(f"Kategori sama: {source.get('category', '')}")

        if color_match > 0.5:
            reasons.append(f"Warna serupa: {source.get('color', '')} ~ {candidate.get('color', '')}")

        if location_sim > 0.3:
            reasons.append(f"Lokasi berdekatan: {source.get('location', '')} ~ {candidate.get('location', '')}")

        return '. '.join(reasons) + '.'

    def get_results(self) -> dict:
        """Return hasil evaluasi model."""
        return self.results or {}
