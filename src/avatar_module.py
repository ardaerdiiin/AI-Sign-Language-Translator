import os
import random
import cv2
from pathlib import Path
from PyQt5.QtCore import QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap

# Proje kök dizini (src/ -> üst dizin)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class AvatarRenderer:
    """
    Kullanıcının girdiği kelimeye karşılık gelen MP4 video dosyasını (Avatar yedeği) bulur.
    Unity'nin bağlı olmadığı durumlarda (veya zorunlu Video Modu aktifken) 'raw_videos' 
    klasörünü tarayarak istenen kelimenin sistemde var olup olmadığını kontrol eder.
    """

    def __init__(self, raw_videos_path='raw_videos'):
        self.raw_videos_path = str(_PROJECT_ROOT / raw_videos_path)
        self.available_words = self._load_words()

    def _load_words(self):
        """raw_videos klasöründeki kelimeleri listeler."""
        if not os.path.exists(self.raw_videos_path):
            print(f"Uyarı: {self.raw_videos_path} bulunamadı.")
            return []
        return [d for d in os.listdir(self.raw_videos_path)
                if os.path.isdir(os.path.join(self.raw_videos_path, d))]

    def get_video_path(self, text):
        """Verilen metne karşılık gelen rastgele bir video yolu döndürür."""
        target_word = text.strip()

        # Tam eşleşme ara (Case insensitive)
        match = None
        for word in self.available_words:
            if word.lower() == target_word.lower():
                match = word
                break

        if match:
            word_path = os.path.join(self.raw_videos_path, match)
            videos = [f for f in os.listdir(word_path)
                      if f.lower().endswith(('.mp4', '.avi', '.mov'))]
            if videos:
                selected_video = random.choice(videos)
                return os.path.join(word_path, selected_video)

        print(f"Kelime bulunamadı: {text}")
        return None


class AvatarPlayer(QObject):
    """
    AvatarRenderer'dan alınan video dosyasını arayüzde (PyQt5 sağ panel) oynatan asenkron video oynatıcı.
    Oynatma işlemi sırasında ana ekranın (UI) donmamasını (freeze) sağlamak için QTimer kullanır.
    Video karelerini 'frame_ready' sinyaliyle UI'a gönderir, böylece kameradan gelen 
    canlı yapay zeka (MediaPipe) analizi arka planda kesintisiz çalışmaya devam edebilir.
    """
    frame_ready = pyqtSignal(QPixmap)
    playback_finished = pyqtSignal()

    def __init__(self, raw_videos_dir=None, fps=30):
        super().__init__()
        if raw_videos_dir is None:
            raw_videos_dir = str(_PROJECT_ROOT / 'raw_videos')
        self.raw_videos_dir = raw_videos_dir
        self.fps = fps
        self.cap = None

        # UI'ı bloklamamak için QTimer kullanıyoruz
        self.timer = QTimer()
        self.timer.timeout.connect(self._next_frame)

    def play_word(self, word):
        """Kelimeye ait videoyu bulur ve oynatmayı başlatır."""
        word_dir = os.path.join(self.raw_videos_dir, word)

        if not os.path.isdir(word_dir):
            print(f"Hata: {word} için video klasörü bulunamadı!")
            self.playback_finished.emit()
            return

        # Klasördeki ilk videoyu al
        video_files = [f for f in os.listdir(word_dir)
                       if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        if not video_files:
            print(f"Hata: {word} klasöründe video yok!")
            self.playback_finished.emit()
            return

        video_path = os.path.join(word_dir, video_files[0])
        self._start_video(video_path)

    def play_path(self, video_path):
        """Doğrudan bir video yolunu oynatır (AvatarRenderer ile birlikte kullanım için)."""
        if not video_path or not os.path.exists(video_path):
            print(f"Hata: Video dosyası bulunamadı -> {video_path}")
            self.playback_finished.emit()
            return
        self._start_video(video_path)

    def _start_video(self, path):
        """Videoyu açar ve QTimer döngüsünü başlatır."""
        self.stop()

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            print(f"Hata: Video açılamadı -> {path}")
            self.playback_finished.emit()
            return

        # QTimer'ı başlat (1000ms / fps = her karenin ekranda kalma süresi)
        self.timer.start(int(1000 / self.fps))

    def _next_frame(self):
        """
        QTimer her tetiklendiğinde (fps hızına göre) sıradaki video karesini okur.
        Okunan kareyi OpenCV BGR formatından PyQt RGB formatına çevirerek 
        arayüze (QLabel) gönderir. Video bittiğinde stop() çağırır.
        """
        if self.cap is None:
            return

        ret, frame = self.cap.read()

        if ret:
            # OpenCV (BGR) formatını PyQt (RGB) formatına çevir
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w

            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)

            # Sinyali tetikle (Ana thread'deki QLabel bunu yakalayacak)
            self.frame_ready.emit(pixmap)
        else:
            # Video bitti
            self.stop()
            self.playback_finished.emit()

    def stop(self):
        """Mevcut oynatmayı güvenli şekilde durdurur."""
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# Test Bloğu
if __name__ == "__main__":
    renderer = AvatarRenderer()
    print("Mevcut Kelimeler:", renderer.available_words)

    test_word = "Merhaba"
    path = renderer.get_video_path(test_word)
    if path:
        print(f"Video yolu: {path}")
    else:
        print("Video bulunamadı.")
