import cv2
import os
import json
import mediapipe as mp
from pathlib import Path

# keypoints_utils modülünü import edebilmek için system path ekle
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.keypoints_utils import mediapipe_detection, extract_raw_landmarks

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_VIDEOS_PATH = _PROJECT_ROOT / 'raw_videos'
REF_LANDMARKS_PATH = _PROJECT_ROOT / 'data' / 'ref_landmarks'
REF_LANDMARKS_PATH.mkdir(parents=True, exist_ok=True)

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands


def extract_for_word(action_name, video_path, pose_model, hands_model):
    cap = cv2.VideoCapture(str(video_path))
    frames_data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        image, results = mediapipe_detection(frame, pose_model, hands_model)
        
        # Filtrelenmiş landmark verisini oluştur
        frame_payload = extract_raw_landmarks(results)
        frames_data.append(frame_payload)
        
    cap.release()
    
    if len(frames_data) > 0:
        out_path = REF_LANDMARKS_PATH / f"{action_name}.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(frames_data, f, ensure_ascii=False)
        print(f"[{action_name}] Referans animasyon kaydedildi: {len(frames_data)} kare -> {out_path.name}")
    else:
        print(f"[{action_name}] UYARI: Video okunamadı veya kare çıkarılamadı.")


def main():
    if not RAW_VIDEOS_PATH.exists():
        print(f"Hata: {RAW_VIDEOS_PATH} bulunamadı.")
        return
        
    actions = [d for d in os.listdir(RAW_VIDEOS_PATH) if os.path.isdir(os.path.join(RAW_VIDEOS_PATH, d))]
    
    if not actions:
        print("Eğitim kelimesi (klasör) bulunamadı.")
        return
        
    print(f"Toplam {len(actions)} kelime için referans landmarklar çıkarılıyor...\n")
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_model, \
         mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands_model:
        for action in actions:
            action_dir = RAW_VIDEOS_PATH / action
            videos = [f for f in os.listdir(action_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            
            if not videos:
                print(f"[{action}] Klasör boş, atlanıyor.")
                continue
                
            # Her kelime için model (altın standart) olarak ilk videoyu seçiyoruz [0]
            video_file = videos[0]
            video_path = action_dir / video_file
            print(f"İşleniyor: {action} (Kaynak: {video_file})")
            
            extract_for_word(action, video_path, pose_model, hands_model)

if __name__ == '__main__':
    main()
