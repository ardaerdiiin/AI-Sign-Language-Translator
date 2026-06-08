import numpy as np
import time

class OneEuroFilter:
    """
    1-Euro Filter: Titreme (jitter) ve sinyal gecikmesini (lag) otomatik ayarlayan matematiksel filtre.
    Yavaş hareketlerde heavy-smoothing (sert yumuşatma), hızlı hareketlerde low-smoothing (hızlı tepki) uygular.
    """
    def __init__(self, min_cutoff=0.05, beta=1.5, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def process(self, t, x):
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x

        t_e = t - self.t_prev
        t_e = max(t_e, 1e-4)

        # Türev
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # Cut-off dinamik hesaplaması (Vektörlerin uzunluğu üzerinden hız bulunur)
        # N boyutlu (Örn: 33 noktalı 3D dizi) için Euclidean hız
        speed = np.linalg.norm(dx_hat, axis=1, keepdims=True)
        
        cutoff = self.min_cutoff + self.beta * speed
        a = self.smoothing_factor(t_e, cutoff)
        
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat


class RobustLandmarkFilter:
    """
    Pose ve Hands verileri için uçtan uca koruma hattı:
    1. Visibility (Görünürlük) Filtresi: Kötü izlenen noktaları dondurur.
    2. 1-Euro Filtresi: X/Y/Z titreşimlerini mükemmel pürüzsüzleştirir.
    3. Z-Smoothing (5 Frame): Kolların ileri/geri (derinlik) zıplamasını yumuşatır.
    """
    def __init__(self, num_points=33, is_pose=True, visibility_threshold=0.5, z_window=5):
        self.num_points = num_points
        self.is_pose = is_pose
        self.vis_threshold = visibility_threshold
        
        # 1-Euro ayarları (Eller daha hızlı, Vücut daha yavaş)
        beta = 0.5 if is_pose else 0.8
        min_cutoff = 0.5 if is_pose else 1.0

        self.one_euro = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=1.0)
        self.z_history = []
        self.z_window = z_window
        
        self.last_valid_pts = np.zeros((num_points, 3))

    def process(self, pts, visibility=None):
        t = time.time()
        pts = np.array(pts, dtype=np.float32)
        
        # 1: Visibility Threshold Kontrolü (Pose için)
        if self.is_pose and visibility is not None:
            vis_arr = np.array(visibility)
            bad_mask = vis_arr < self.vis_threshold
            
            # Kötü tespit edilmişse, o noktayı son bilinen "iyi" konumda DONDUR
            pts[bad_mask] = self.last_valid_pts[bad_mask]
            
            # İyi tespitlerin son konumunu kaydet
            good_mask = ~bad_mask
            if np.any(good_mask):
                self.last_valid_pts[good_mask] = pts[good_mask]
        else:
            # Sadece eller veya visbility verilmeyen durumlar
            self.last_valid_pts = pts.copy()

        # 2: One-Euro 3D Filtresi 
        filtered_pts = self.one_euro.process(t, pts)
        
        # 3: Z Eksenine Özel Kayan Ortalama (Rolling Z-Smooth)
        self.z_history.append(filtered_pts[:, 2].copy())
        if len(self.z_history) > self.z_window:
            self.z_history.pop(0)
            
        smoothed_z = np.mean(self.z_history, axis=0)
        filtered_pts[:, 2] = smoothed_z
        
        return filtered_pts.tolist()
