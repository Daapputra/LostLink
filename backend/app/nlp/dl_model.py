"""
LostLink Deep Learning Model Module
=====================================
Modul untuk loading dan penggunaan model Word Embedding + LSTM
untuk klasifikasi kategori dan similarity computation.
"""

import os
import json
import pickle
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'trained_models')


class DLModel:
    """Wrapper untuk model Deep Learning (Embedding + LSTM)."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.max_len = 100
        self.results = None
        self.is_loaded = False

    def load(self):
        """Muat model dari disk."""
        try:
            model_path = os.path.join(MODEL_DIR, 'lstm_model.keras')
            tokenizer_path = os.path.join(MODEL_DIR, 'tokenizer.pickle')
            config_path = os.path.join(MODEL_DIR, 'tokenizer_config.json')

            if not os.path.exists(model_path):
                print("[LSTM] Model file not found.")
                return False

            # Import TensorFlow
            import tensorflow as tf
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

            self.model = tf.keras.models.load_model(model_path)

            # Load tokenizer
            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)
            else:
                print("[LSTM] Tokenizer not found.")
                return False

            # Load config
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.max_len = config.get('max_len', 100)

            # Load label encoder
            import joblib
            encoder_path = os.path.join(MODEL_DIR, 'label_encoder.joblib')
            if os.path.exists(encoder_path):
                self.label_encoder = joblib.load(encoder_path)

            # Load evaluation results
            results_path = os.path.join(MODEL_DIR, 'lstm_results.json')
            if os.path.exists(results_path):
                with open(results_path, 'r', encoding='utf-8') as f:
                    self.results = json.load(f)

            self.is_loaded = True
            print("[LSTM] Model loaded successfully.")
            return True

        except Exception as e:
            print(f"[LSTM] Error loading model: {e}")
            return False

    def predict_category(self, text: str) -> dict:
        """
        Prediksi kategori barang dari deskripsi teks menggunakan LSTM.

        Returns:
            dict: {'category': str, 'confidence': float, 'probabilities': dict}
        """
        if not self.is_loaded:
            return {'error': 'Model not loaded'}

        from .preprocessing import preprocess_for_embedding
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        clean_text = preprocess_for_embedding(text)

        # Tokenize dan pad
        sequence = self.tokenizer.texts_to_sequences([clean_text])
        padded = pad_sequences(sequence, maxlen=self.max_len, padding='post', truncating='post')

        # Prediksi
        prediction = self.model.predict(padded, verbose=0)[0]
        predicted_idx = int(np.argmax(prediction))
        confidence = float(prediction[predicted_idx])

        category = self.label_encoder.inverse_transform([predicted_idx])[0]

        # Semua probabilitas
        all_probs = {}
        for i, cat in enumerate(self.label_encoder.classes_):
            all_probs[cat] = round(float(prediction[i]), 4)

        return {
            'category': category,
            'confidence': round(confidence, 4),
            'probabilities': all_probs
        }

    def compute_embedding_similarity(self, text1: str, text2: str) -> float:
        """
        Hitung similarity menggunakan embedding representation.
        Mengambil output dari layer sebelum output untuk mendapatkan representasi vektor.
        """
        if not self.is_loaded:
            return 0.0

        try:
            import tensorflow as tf
            from .preprocessing import preprocess_for_embedding
            from tensorflow.keras.preprocessing.sequence import pad_sequences

            clean1 = preprocess_for_embedding(text1)
            clean2 = preprocess_for_embedding(text2)

            seq1 = self.tokenizer.texts_to_sequences([clean1])
            seq2 = self.tokenizer.texts_to_sequences([clean2])

            pad1 = pad_sequences(seq1, maxlen=self.max_len, padding='post', truncating='post')
            pad2 = pad_sequences(seq2, maxlen=self.max_len, padding='post', truncating='post')

            # Dapatkan representasi dari layer sebelum output
            intermediate_model = tf.keras.Model(
                inputs=self.model.input,
                outputs=self.model.layers[-2].output  # Layer Dense sebelum output
            )

            vec1 = intermediate_model.predict(pad1, verbose=0)
            vec2 = intermediate_model.predict(pad2, verbose=0)

            # Cosine similarity
            dot = np.dot(vec1[0], vec2[0])
            norm1 = np.linalg.norm(vec1[0])
            norm2 = np.linalg.norm(vec2[0])

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(dot / (norm1 * norm2))

        except Exception as e:
            print(f"[LSTM] Embedding similarity error: {e}")
            return 0.0

    def get_results(self) -> dict:
        """Return hasil evaluasi model."""
        return self.results or {}
