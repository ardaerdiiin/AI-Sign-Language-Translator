"""
Keypoint extraction ve MediaPipe detection fonksiyonları.
Tek Gerçeklik Kaynağı (Single Source of Truth):
  - process_videos.py ve inference.py bu modülü kullanır.
  - Özellik vektörü: 194 boyut (üst_gövde:24 + el_koord:120 + açılar:30 + mesafeler:20)
  - Feature Engineering: bilek merkezli normalizasyon + eklem açıları + parmak ucu mesafeleri
"""

import cv2
import numpy as np

# Filtreleri yükle
from src.filters import RobustLandmarkFilter

# Filtre objelerini global veya sınıf seviyesi olarak saklıyoruz 
# (Her kare the stateful OneEuroFilter için)
pose_filter = RobustLandmarkFilter(num_points=33, is_pose=True, visibility_threshold=0.5, z_window=5)
lh_filter = RobustLandmarkFilter(num_points=21, is_pose=False, z_window=5)
rh_filter = RobustLandmarkFilter(num_points=21, is_pose=False, z_window=5)

class UnifiedResults:
    def __init__(self):
        self.pose_landmarks = None
        self.left_hand_landmarks = None
        self.right_hand_landmarks = None

def mediapipe_detection(image, pose_model, hands_model):
    """
    Kameradan gelen görüntüyü alarak MediaPipe modelleriyle vücut ve elleri analiz eder.
    NOT: MediaPipe'ın ayna efekti nedeniyle ekranda solda görünen elin 'Sağ' olarak etiketlenmesini
    düzelterek (anatomik doğruluk) tek bir objede (UnifiedResults) döndürür.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    pose_results = pose_model.process(image)
    hands_results = hands_model.process(image)
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    results = UnifiedResults()
    results.pose_landmarks = pose_results.pose_landmarks
    
    # MediaPipe Hands için Ayna(Label) Düzeltmesi (ÇÜNKÜ HOLISTIC'E GÖRE TERS ÇALIŞIR)
    if hands_results.multi_hand_landmarks:
        for idx, hand_handedness in enumerate(hands_results.multi_handedness):
            label = hand_handedness.classification[0].label
            # Ekranda (önden) solda duran el anatomik olarak SAĞ eldir!
            if label == 'Left':
                results.right_hand_landmarks = hands_results.multi_hand_landmarks[idx]
            else:
                results.left_hand_landmarks = hands_results.multi_hand_landmarks[idx]
                
    return image, results


# ============================================================================
# Feature Engineering Sabitleri & Yardımcı Fonksiyonlar
# ============================================================================

EPS = 1e-7  # Sıfıra bölme koruması

# Üst gövde landmark indeksleri (pose 33 noktadan seçim)
# 11=sol omuz, 12=sağ omuz, 13=sol dirsek, 14=sağ dirsek,
# 15=sol bilek, 16=sağ bilek, 23=sol kalça, 24=sağ kalça
_UPPER_BODY_INDICES = np.array([11, 12, 13, 14, 15, 16, 23, 24])

# Parmak eklem zincirleri (her parmak 4 landmark, 3 açı üretir)
# Her tuple: (proksimal, orta, distal, uç)
_FINGER_CHAINS = np.array([
    [1, 2, 3, 4],      # Başparmak
    [5, 6, 7, 8],      # İşaret parmağı
    [9, 10, 11, 12],   # Orta parmak
    [13, 14, 15, 16],  # Yüzük parmağı
    [17, 18, 19, 20],  # Serçe parmak
])

# Parmak uçları indeksleri (mesafe hesabı için)
_FINGERTIP_INDICES = np.array([4, 8, 12, 16, 20])


def _compute_joint_angles(hand: np.ndarray) -> np.ndarray:
    """
    Elin (21 nokta) her parmağı için 3 ana eklem açısını hesaplar. Toplamda 15 açı üretir.
    Bu hesaplama, elin kameraya yakınlığına veya ekran konumuna bağlı olmadığı için
    modele çok daha stabil (invariant) bir özellik seti sunar.

    Her parmakta A-B-C üçlüsünden bükülme açısı:
      angle = arccos(dot(BA, BC) / (|BA|·|BC| + ε)) / π   → [0, 1] aralığına normalize.

    Args:
        hand: (21, 3) boyutunda el landmark dizisi.
    Returns:
        (15,) boyutunda açı vektörü.
    """
    # El tamamen sıfır → algılanmamış, güvenli sıfır dön
    if np.all(hand == 0):
        return np.zeros(15, dtype=np.float32)

    # Her parmak için 3 açı noktası (A-B-C üçlülerini oluştur)
    # Zincir [p0, p1, p2, p3] → açılar: (p0,p1,p2), (p1,p2,p3)
    # Ama 4 noktadan 3 açı: (p0-p1-p2), (p1-p2-p3), ve ek olarak bileğe bağlantı
    # Standart: 4 nokta → 2 iç açı + 1 bilek açısı
    # Basitleştirilmiş: Her zincirden 3 ardışık üçlü

    # A noktaları: zincirin 0,1,2. elemanları → (5×3 = 15 üçlü)
    # B noktaları: zincirin 1,2,3. elemanları
    # Ama 4 noktadan sadece 2 açı çıkar (3 ardışık nokta gerek)
    # chains[:, 0:3] → ilk üçlü, chains[:, 1:4] → ikinci üçlü
    # + bilek(0) ile ilk eklem arasındaki açı

    # Bilek-tabanlı açı: (bilek=0, chain[0], chain[1])
    wrist = hand[0:1]  # (1, 3)
    a0 = np.tile(wrist, (5, 1))               # 5 parmak için bilek tekrarı
    b0 = hand[_FINGER_CHAINS[:, 0]]            # (5, 3) — her parmağın tabanı
    c0 = hand[_FINGER_CHAINS[:, 1]]            # (5, 3) — MCP eklemi

    # İç eklem açıları: (chain[0]-chain[1]-chain[2]) ve (chain[1]-chain[2]-chain[3])
    a1 = hand[_FINGER_CHAINS[:, 0]]
    b1 = hand[_FINGER_CHAINS[:, 1]]
    c1 = hand[_FINGER_CHAINS[:, 2]]

    a2 = hand[_FINGER_CHAINS[:, 1]]
    b2 = hand[_FINGER_CHAINS[:, 2]]
    c2 = hand[_FINGER_CHAINS[:, 3]]

    # Hepsini birleştir → (15, 3) boyutunda A, B, C dizileri
    A = np.concatenate([a0, a1, a2], axis=0)   # (15, 3)
    B = np.concatenate([b0, b1, b2], axis=0)   # (15, 3)
    C = np.concatenate([c0, c1, c2], axis=0)   # (15, 3)

    # Vektör hesabı (tamamen vektörize)
    v1 = A - B  # (15, 3)
    v2 = C - B  # (15, 3)

    # Dot product (satır bazında)
    dot = np.sum(v1 * v2, axis=1)              # (15,)
    norms = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + EPS
    cos_angles = np.clip(dot / norms, -1.0, 1.0)

    angles = np.arccos(cos_angles) / np.pi     # [0, 1] normalize

    return angles.astype(np.float32)


def _compute_fingertip_distances(hand: np.ndarray) -> np.ndarray:
    """
    5 parmak ucunun birbirine olan mesafelerini hesaplar.
    C(5,2) = 10 çift.

    Args:
        hand: (21, 3) el landmark dizisi.

    Returns:
        (10,) mesafe vektörü.
    """
    if np.all(hand == 0):
        return np.zeros(10, dtype=np.float32)

    tips = hand[_FINGERTIP_INDICES]  # (5, 3)

    # Çift indeksleri (upper triangle) — compile-time sabit
    # (0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
    i_idx = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3])
    j_idx = np.array([1, 2, 3, 4, 2, 3, 4, 3, 4, 4])

    diffs = tips[i_idx] - tips[j_idx]          # (10, 3)
    distances = np.linalg.norm(diffs, axis=1)  # (10,)

    return distances.astype(np.float32)


def _center_hand_on_wrist(hand: np.ndarray) -> np.ndarray:
    """
    Elin tüm koordinatlarını (21 nokta), 0. indeks olan 'Bilek' (Wrist) noktasına göre yeniden konumlandırır.
    Bu işlem (lokal normalizasyon) sayesinde eller ekranın neresinde olursa olsun koordinat dizilimi aynı kalır.

    Args:
        hand: (21, 3) el landmark dizisi.
    Returns:
        (20, 3) bilek-göreli koordinatlar (bilek noktası hariç).
    """
    if np.all(hand == 0):
        return np.zeros((20, 3), dtype=np.float32)

    wrist = hand[0]              # (3,)
    centered = hand[1:] - wrist  # (20, 3)

    return centered.astype(np.float32)


# ============================================================================
# Ana Keypoint Çıkarım Fonksiyonu
# ============================================================================

def extract_keypoints(results):
    """
    MediaPipe sonuçlarını alır ve yapay zeka modelinin giriş şekline (194 boyutlu özellik vektörü) dönüştürür.
    Hem eğitimde (process_videos) hem de canlı tahminde (inference) standart veri üreten kritik fonksiyondur.

    Süreç:
      1. Ham koordinatları al.
      2. Gövdeyi burna göre merkezle, omuz genişliğine bölerek ölçekle (kamera mesafesinden bağımsız).
      3. Elleri bileklere göre merkezle (lokal normalizasyon).
      4. Eklem açılarını hesapla (15+15 = 30).
      5. Parmak ucu mesafelerini hesapla (10+10 = 20).

    Returns:
        np.ndarray: 194 boyutlu özellik vektörü.
    """
    # --- Ham verileri al ---
    pose = (
        np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark])
        if results.pose_landmarks
        else np.zeros((33, 4))
    )
    lh = (
        np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark])
        if results.left_hand_landmarks
        else np.zeros((21, 3))
    )
    rh = (
        np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])
        if results.right_hand_landmarks
        else np.zeros((21, 3))
    )

    # --- Pose: Merkezileştirme & Ölçekleme ---
    if results.pose_landmarks:
        # 1. Burun merkezleme
        ref = pose[0, :3]  # (3,)
        pose[:, :3] -= ref

        # 2. Omuz genişliği ölçekleme
        shoulder_left = pose[11, :3]
        shoulder_right = pose[12, :3]
        width = np.linalg.norm(shoulder_left - shoulder_right)

        if width > 0.1:
            pose[:, :3] /= width

    # Sadece üst gövde noktaları (8 nokta × 3 koordinat = 24)
    upper_body = pose[_UPPER_BODY_INDICES, :3].flatten()  # (24,)

    # --- Eller: Lokal Normalizasyon (li) ---
    lh_centered = _center_hand_on_wrist(lh)   # (20, 3)
    rh_centered = _center_hand_on_wrist(rh)   # (20, 3)


    # NOT: Bilek merkezli koordinatlar zaten kameradan bağımsız (görelidir).
    # Açılar ve mesafeler ölçekten bağımsız olduğu için ek ölçekleme gerekmez.


    # --- Açılar (15 + 15 = 30) ---
    lh_angles = _compute_joint_angles(lh)  # (15,)
    rh_angles = _compute_joint_angles(rh)  # (15,)

    # --- Parmak uçları mesafeleri (10 + 10 = 20) ---
    lh_distances = _compute_fingertip_distances(lh)  # (10,)
    rh_distances = _compute_fingertip_distances(rh)  # (10,)

    # --- Birleştir: 24 + 60 + 60 + 15 + 15 + 10 + 10 = 194 ---
    return np.concatenate([
        upper_body,              # (24,) üst gövde
        lh_centered.flatten(),   # (60,) sol el bilek-göreli
        rh_centered.flatten(),   # (60,) sağ el bilek-göreli
        lh_angles,               # (15,) sol el eklem açıları
        rh_angles,               # (15,) sağ el eklem açıları
        lh_distances,            # (10,) sol parmak ucu mesafeleri
        rh_distances,            # (10,) sağ parmak ucu mesafeleri
    ])


def extract_raw_landmarks(results):
    """
    Avatar kontrolü (C# Unity) için MediaPipe Pose + Hands sonuçlarından HAM koordinatları çıkarır.
    İşaret dili modeline değil, Unity'deki Avatar IK yöneticisine veri sağlamak amacıyla kullanılır.
    Ek olarak 'One Euro Filter' ile koordinat titremelerini (jittering) yumuşatır.
    """
    has_lh = results.left_hand_landmarks is not None
    has_rh = results.right_hand_landmarks is not None

    if results.pose_landmarks:
        pose_raw = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
        pose_vis = [lm.visibility for lm in results.pose_landmarks.landmark]
        pose = pose_filter.process(pose_raw, visibility=pose_vis)
    else:
        pose = [[0.0, 0.0, 0.0]] * 33

    if has_lh:
        lh_raw = [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
        lh = lh_filter.process(lh_raw)
    else:
        lh = [[0.0, 0.0, 0.0]] * 21

    if has_rh:
        rh_raw = [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
        rh = rh_filter.process(rh_raw)
    else:
        rh = [[0.0, 0.0, 0.0]] * 21

    return {
        "type": "L",
        "pose": pose,
        "lh": lh,
        "rh": rh,
        "has_lh": has_lh,
        "has_rh": has_rh,
    }
