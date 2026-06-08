import cv2
import numpy as np
import os
import json
import mediapipe as mp
from pathlib import Path

from src.keypoints_utils import mediapipe_detection, extract_keypoints, extract_raw_landmarks

# --- Yapılandırma (Mutlak Yollar) ---
_PROJECT_ROOT = Path(__file__).resolve().parent
RAW_VIDEOS_PATH = str(_PROJECT_ROOT / 'raw_videos')
DATA_PATH = str(_PROJECT_ROOT / 'data')
LABELS_PATH = str(_PROJECT_ROOT / 'labels.json')
REF_LANDMARKS_PATH = str(_PROJECT_ROOT / 'data' / 'ref_landmarks')
NEW_RAW_VIDEOS_PATH = str(_PROJECT_ROOT / 'new_raw_videos')
SEQUENCE_LENGTH = 16  # Transformer Modelin beklediği kare sayısı (TSLFormer ile aynı)

# --- MediaPipe Kurulumu ---
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands


def process_video(video_path, action_name, sequence_num, pose_model, hands_model):
    """
    Tek bir MP4 videosunu alıp yapay zekanın eğitebileceği formata (NumPy .npy dizileri) dönüştürür.
    Videoyu baştan sona okur, iskelet (landmark) noktalarını çıkarır.
    Video uzunluğu ne olursa olsun, zaman ekseninde homojen (Uniform Sampling) dağılmış tam 16 kare seçer.
    Ekstra Özellik: Kelimenin ilk videosunu işlerken, Unity Avatarının canlandırabilmesi için 
    hareketin tüm karelerini JSON dosyası (Referans Animasyon) olarak kaydeder.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print(f"Hata: {video_path} okunamadi veya bos.")
        return False

    if total_frames < 10:
        print(f"Uyari: {video_path} çok kisa ({total_frames} kare), atlaniyor.")
        return False

    # Uniform Sampling: hangi karelerin seçileceğini hesapla
    indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=int)

    sequence_path = os.path.join(DATA_PATH, action_name, str(sequence_num))
    if not os.path.exists(sequence_path):
        os.makedirs(sequence_path)

    # Seçili kareleri LSTM için işle
    extracted_frames = []
    
    # 1. Video (Seq 0) ise Avatar Animasyonu (JSON) için tüm kareleri 3B RAW formatında kaydet!
    raw_animation_frames = []
    is_first_video = (sequence_num == 0)

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        # MediaPipe işleniyor
        image, results = mediapipe_detection(frame, pose_model, hands_model)
        
        # Sadece seçili indeksler LSTM .npy verisi için ayrılır
        if i in indices:
            keypoints = extract_keypoints(results)
            extracted_frames.append(keypoints)
            
        # Eğer kelimenin ilk videosuysa, Unity Avatar'ı için pürüzsüz animasyon referansı (JSON) topla
        if is_first_video:
            raw_lm = extract_raw_landmarks(results)
            raw_animation_frames.append(raw_lm)

    cap.release()

    # Eksik kare varsa son kareyle tamamla
    while len(extracted_frames) < SEQUENCE_LENGTH:
        extracted_frames.append(extracted_frames[-1])

    # Fazla varsa kırp
    extracted_frames = extracted_frames[:SEQUENCE_LENGTH]

    # Kaydet (LSTM NumPy Dosyaları)
    for i, keys in enumerate(extracted_frames):
        npy_path = os.path.join(sequence_path, f"{i}.npy")
        np.save(npy_path, keys)

    # Kaydet (Unity Avatar JSON) -> Sadece ilk videoda üretilir
    if is_first_video and len(raw_animation_frames) > 0:
        if not os.path.exists(REF_LANDMARKS_PATH):
            os.makedirs(REF_LANDMARKS_PATH)
        json_path = os.path.join(REF_LANDMARKS_PATH, f"{action_name}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(raw_animation_frames, f, ensure_ascii=False)
        print(f"      -> {action_name}.json referans animasyonu olusturuldu! ({len(raw_animation_frames)} kare)")

    return True


def _update_labels_json(actions):
    """
    Kelimelerin sayısal ID'lere eşleştirildiği `labels.json` dosyasını yönetir.
    Yeni eklenen kelime klasörleri varsa, mevcut indeks sırasını bozmadan yeni ID'ler atar.
    Model, kelimenin metnini değil, buradaki ID numarasını (Örn: "Merhaba" -> 0, "Nasılsın" -> 1) tahmin eder.
    """
    # Mevcut labels.json varsa oku, yoksa boş dict
    label_map = {}
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, 'r', encoding='utf-8') as f:
            label_map = json.load(f)

    # Yeni kelimeleri ekle (mevcut indeksleri bozmadan)
    next_idx = max(label_map.values()) + 1 if label_map else 0
    for action in sorted(actions):
        if action not in label_map:
            label_map[action] = next_idx
            next_idx += 1
            print(f"  Yeni etiket eklendi: {action} -> {label_map[action]}")

    # Kaydet
    with open(LABELS_PATH, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print(f"labels.json güncellendi ({len(label_map)} etiket).")


def main():
    """
    Veri Hazırlık Sürecinin (Data Preprocessing) Ana Döngüsü.
    - 'raw_videos' dizinini tarayarak işaret dili videolarını (mp4 vb.) bulur.
    - MediaPipe modellerini başlatır.
    - Tüm videoları sırayla `process_video` fonksiyonundan geçirerek LSTM eğitim verisi olan `.npy` dosyalarını oluşturur.
    - İşlem bitince `labels.json` dosyasını günceller.
    (Modeli yeniden eğitmeden önce her zaman bu dosya çalıştırılmalıdır).
    """
    source_path = RAW_VIDEOS_PATH
    
    has_new_videos = False
    if os.path.exists(NEW_RAW_VIDEOS_PATH):
        if len(os.listdir(NEW_RAW_VIDEOS_PATH)) > 0:
            has_new_videos = True

    if has_new_videos:
        print("Sistemde 'new_raw_videos' klasoru bulundu ve icinde veriler var.")
        print("Sadece yeni videolar islenecek.")
        source_path = NEW_RAW_VIDEOS_PATH
    else:
        print("'new_raw_videos' klasoru bos veya bulunamadi.")
        ans = input("Tum veri setini ('raw_videos') bastan asagi islemeye devam etmek istediginize emin misiniz? (y/n): ")
        if ans.lower() != 'y':
            print("Islem iptal edildi.")
            return
        if not os.path.exists(RAW_VIDEOS_PATH):
            print(f"Klasor bulunamadi: {RAW_VIDEOS_PATH}")
            print("Lutfen videolari 'raw_videos/KELIME_ADI' seklinde yerlestirin.")
            os.makedirs(RAW_VIDEOS_PATH)
            return

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_model, \
         mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands_model:
         
        actions = [d for d in os.listdir(source_path)
                   if os.path.isdir(os.path.join(source_path, d))]

        for action in actions:
            action_dir = os.path.join(source_path, action)

            print(f"Islemiyor: {action}...")
            videos = [f for f in os.listdir(action_dir)
                      if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                      
            # Sequence numarasini mevcut verilerden devam ettir (Uzerine yazmayi engelle)
            target_data_dir = os.path.join(DATA_PATH, action)
            next_seq_num = 0
            if os.path.exists(target_data_dir):
                existing_seqs = [int(d) for d in os.listdir(target_data_dir) if d.isdigit()]
                if existing_seqs:
                    next_seq_num = max(existing_seqs) + 1

            for i, video_file in enumerate(videos):
                video_path = os.path.join(action_dir, video_file)
                current_seq_num = next_seq_num + i
                print(f"  -> Video: {video_file} (Seq: {current_seq_num})")

                success = process_video(video_path, action, current_seq_num, pose_model, hands_model)
                if success:
                    print(f"     Tamamlandi.")
                else:
                    print(f"     Basarisiz.")

        # labels.json güncelle
        _update_labels_json(actions)


if __name__ == '__main__':
    main()
