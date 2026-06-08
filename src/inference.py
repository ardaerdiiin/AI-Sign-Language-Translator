import cv2
import numpy as np
import os
import json
import mediapipe as mp
from pathlib import Path
from collections import deque

from src.keypoints_utils import mediapipe_detection, extract_keypoints

# Proje kök dizini (src/ -> üst dizin)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class SignLanguagePredictor:
    """
    Gerçek zamanlı kamera görüntülerinden işaret dili tahmini yapan ana motor.
    MediaPipe ile vücut/el referans (landmark) noktalarını çıkarır, hareketleri 
    belirli bir kare uzunluğunda (sekans) toplar ve LSTM (veya TFLite) modeli 
    üzerinden geçirerek tahmin sonuçlarını üretir.
    Ayrıca spam engelleme (cooldown) ve tahmin filtreleme gibi mekanizmaları içerir.
    """
    def __init__(self, model_path=None, labels_path=None):
        if model_path is None:
            model_path = str(_PROJECT_ROOT / 'action')  # Uzantısız: .tflite veya .h5 aranacak
        if labels_path is None:
            labels_path = str(_PROJECT_ROOT / 'labels.json')

        self.actions = self._load_actions(labels_path)

        # --- Model Yükleme: TFLite öncelikli, H5 fallback ---
        self.use_tflite = False
        self._tflite_interpreter = None
        self._tflite_input_details = None
        self._tflite_output_details = None
        self.model = None

        tflite_path = model_path + '.tflite'
        h5_path = model_path + '.h5'

        if os.path.exists(tflite_path):
            self._load_tflite(tflite_path)
        elif os.path.exists(h5_path):
            self._load_keras(h5_path)
        else:
            print("Model dosyası bulunamadı! (.tflite veya .h5)")

        # MediaPipe (Ayrılmış Pose ve Hands)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2
        )

        # --- Tahmin Degiskenleri (Yumusak Sistem + Margin Filtresi) ---
        self.sequence = []                          # Son 16 karenin keypoint'leri (downsampled)
        self.raw_sequence = []                      # Kayan penceredeki ham keypoint'ler (maks 45 kare)
        self.sentence = []                          # Tahmin edilen kelimeler
        self.prob_buffer = deque(maxlen=15)          # Son 15 karenin olasilik vektorleri
        self.threshold = 0.40                       # Ortalama olasilik esigi (Hassas algilama icin esnetildi)
        self.margin_threshold = 0.05                # Top-1 ve Top-2 arasi min fark (Daha kararli tahmin tetiklemesi icin dusuruldu)
        self.last_prediction = None                 # Son kabul edilen tahmin
        self.cooldown_counter = 0                   # Ayni kelimeyi spam'i onle
        self.COOLDOWN_FRAMES = 15                   # Bir tahmin sonrasi 15 kare bekle

        # --- Gelişmiş Filtreleme ve Downsampling ---
        self.no_hand_frames_counter = 0             # Geçici el kayıplarını tolere etmek için sayaç
        self.blacklisted_actions = set()            # Geçici tahmin kara listesi (yanlış kelimeler için)
        self.prev_has_hands = False                 # Bir önceki karede el var mıydı?

    def blacklist_word(self, word):
        """
        Kullanıcının yanlış bulduğu tahmini geçici kara listeye (blacklist) ekler.
        Böylece model, kullanıcının düzelttiği kelimeyi kısa süre içinde tekrar tahmin etmez.
        """
        formatted_word = word.strip().capitalize()
        self.blacklisted_actions.add(formatted_word)
        print(f"[Blacklist] Kelime geçici olarak engellendi: {formatted_word}")

    def clear_blacklist(self):
        """
        Doğru bir tahmin yapıldıktan sonra geçici kara listeyi sıfırlar.
        Böylece engellenmiş kelimeler yeni hareketler için tekrar tahmin edilebilir hale gelir.
        """
        self.blacklisted_actions.clear()
        print("[Blacklist] Kara liste temizlendi.")

    def _load_actions(self, labels_path):
        """labels.json'dan deterministik etiket sırasını okur."""
        if not os.path.exists(labels_path):
            print(f"Uyarı: {labels_path} bulunamadı! Boş etiket listesi.")
            return np.array([])
        with open(labels_path, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
        # İndekse göre sırala → dizi
        sorted_labels = sorted(label_map.items(), key=lambda x: x[1])
        return np.array([name for name, _ in sorted_labels])

    def _load_tflite(self, path):
        """TFLite modelini yükler — ~7x daha hızlı inference."""
        try:
            import tensorflow as tf
            self._tflite_interpreter = tf.lite.Interpreter(model_path=path)
            self._tflite_interpreter.allocate_tensors()
            self._tflite_input_details = self._tflite_interpreter.get_input_details()
            self._tflite_output_details = self._tflite_interpreter.get_output_details()
            self.use_tflite = True
            print(f"TFLite modeli yuklendi: {path} (hizli inference)")
        except Exception as e:
            print(f"TFLite yukleme hatasi: {e}")
            # Fallback: H5 dene
            h5_path = path.replace('.tflite', '.h5')
            if os.path.exists(h5_path):
                self._load_keras(h5_path)

    def _load_keras(self, path):
        """Keras H5 modelini yükler (fallback)."""
        try:
            from tensorflow.keras.models import load_model
            self.model = load_model(path)
            self.use_tflite = False
            print(f"Keras modeli yuklendi: {path} (standart inference)")
        except Exception as e:
            print(f"Keras yukleme hatasi: {e}")

    def _predict_model(self, input_data):
        """
        Oluşturulan 16 karelik veri dizisini (sekans) modele sokarak tahmin olasılıklarını döndürür.
        Performansı artırmak için öncelikli olarak TFLite modelini (~3ms) tercih eder. 
        TFLite bulunamazsa, standart Keras modelini (~20ms) kullanır.
        """
        if self.use_tflite and self._tflite_interpreter:
            # TFLite: ~3ms
            input_data = np.array(input_data, dtype=np.float32)
            self._tflite_interpreter.set_tensor(
                self._tflite_input_details[0]['index'], input_data
            )
            self._tflite_interpreter.invoke()
            return self._tflite_interpreter.get_tensor(
                self._tflite_output_details[0]['index']
            )[0]
        elif self.model:
            # Keras: ~20ms
            return self.model.predict(input_data, verbose=0)[0]
        return None

    def draw_styled_landmarks(self, image, results):
        self.mp_drawing.draw_landmarks(
            image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
        )
        self.mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
        )
        self.mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )
    def predict(self, image, enable_prediction=True):
        """
        Ana işlem fonksiyonu. Her yeni kare (frame) geldiğinde çalışır:
        1. MediaPipe ile eklemlerin koordinatlarını bulur ve ekrana çizer.
        2. Elin ekranda olup olmadığını kontrol eder (elin inmesi/çıkması mantığı).
        3. Yeterli sayıda kare biriktiğinde modeli çalıştırıp olasılıkları (prob) çıkarır.
        4. Olasılıklar, eşik değerlerini (threshold ve margin) aşarsa tahmini onaylar.
        Geriye: İşlenmiş resim, mediapipe sonuçları, tahmin edilen kelime ve olasılık yüzdesi döner.
        """
        # Webcam selfie modundan AUTSL perspektifine cevir (egitim verisiyle uyumlu)
        image = cv2.flip(image, 1)
        image, results = mediapipe_detection(image, self.pose, self.hands)
        # self.draw_styled_landmarks(image, results)

        predicted_action = None
        prob = 0.0

        if not enable_prediction:
            self.prob_buffer.clear()
            self.cooldown_counter = 0
            self.no_hand_frames_counter = 0
            self.raw_sequence = []
            self.sequence = []
            self.prev_has_hands = False
            return image, results, predicted_action, prob

        # --- EL KONTROLU: Geçici el kayıplarını tolere et (Grace Period) ---
        has_hands = (results.left_hand_landmarks is not None) or (results.right_hand_landmarks is not None)
        
        # Elin havadan indirilip indirilmediğini tespit et (pasif tetikleme sınırı)
        hand_lowered_trigger = self.prev_has_hands and not has_hands
        self.prev_has_hands = has_hands

        if has_hands:
            self.no_hand_frames_counter = 0
        else:
            self.no_hand_frames_counter += 1

        # Eğer el gerçekten uzun süre yoksa (örn: 15 kare, ~0.5 saniye), buffer'ları ve diziyi temizle
        # AMA elin tam indirildiği kare ise (hand_lowered_trigger), son tahminin yapılmasına izin ver.
        if self.no_hand_frames_counter > 15:
            self.prob_buffer.clear()
            self.cooldown_counter = 0
            self.raw_sequence = []
            self.sequence = []
            return image, results, None, 0.0

        # --- SLIDING WINDOW COLLECTOR ---
        if has_hands:
            keypoints = extract_keypoints(results)
            self.raw_sequence.append(keypoints)
            self.raw_sequence = self.raw_sequence[-45:]  # Maks 45 kare (~1.5s)

        # --- TAHMİN KOŞULLARI ---
        should_predict = False
        if has_hands and len(self.raw_sequence) >= 16:
            should_predict = True
        elif hand_lowered_trigger and len(self.raw_sequence) >= 15:
            should_predict = True

        if should_predict:
            indices = np.linspace(0, len(self.raw_sequence) - 1, 16, dtype=int)
            self.sequence = [self.raw_sequence[i] for i in indices]

            if (self.use_tflite or self.model):
                res = self._predict_model(np.expand_dims(self.sequence, axis=0))

                # Olasilik vektorunu buffer'a ekle
                self.prob_buffer.append(res)

                # Ortalama olasilik hesapla (yumusak karar)
                if len(self.prob_buffer) >= 3:
                    avg_probs = np.mean(self.prob_buffer, axis=0)

                    # Kara listedeki kelimelerin olasılığını sıfırla (ez)
                    if hasattr(self, 'blacklisted_actions') and self.blacklisted_actions:
                        for word in self.blacklisted_actions:
                            if word in self.actions:
                                idx = np.where(self.actions == word)[0][0]
                                avg_probs[idx] = -1.0  # Negatif yaparak seçilmesini önle

                    best_idx = np.argmax(avg_probs)
                    prob = avg_probs[best_idx]

                    # Margin hesapla: Top-1 ve Top-2 arasindaki fark
                    top2_indices = avg_probs.argsort()[-2:][::-1]
                    margin = avg_probs[top2_indices[0]] - avg_probs[top2_indices[1]]

                    # DEBUG: konsola tahmin detaylarını yaz
                    top_3_indices = avg_probs.argsort()[-3:][::-1]
                    debug_str = " | ".join(
                        [f"{self.actions[i]}: %{int(avg_probs[i] * 100)}" for i in top_3_indices]
                    )
                    print(f"Ort. Tahmin: {debug_str} | Active: {has_hands} | Trigger: {hand_lowered_trigger}")

                    # --- ÇİFT TETİKLEME EŞİKLERİ ---
                    if has_hands:
                        # 1. Havada Aktif Tetikleme: Erken/yanlış algılamayı önlemek için çok yüksek güven ve min kare isteriz
                        threshold_ok = (prob > 0.75) and (margin > 0.15) and (len(self.raw_sequence) >= 28)
                    else:
                        # 2. El İndirildiğinde Pasif Tetikleme: Hareket bittiği için standart/esnek eşikler yeterlidir
                        threshold_ok = (prob > self.threshold) and (margin > self.margin_threshold)

                    if self.cooldown_counter > 0:
                        self.cooldown_counter -= 1
                    else:
                        if threshold_ok:
                            predicted_action = self.actions[best_idx]
                            self.last_prediction = predicted_action
                            self.cooldown_counter = self.COOLDOWN_FRAMES
                            self.prob_buffer.clear()
                            self.raw_sequence = []  # Tahmin sonrası pencereyi sıfırla

        return image, results, predicted_action, prob

