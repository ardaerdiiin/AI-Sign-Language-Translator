import sys
import cv2
import numpy as np
import time
import socket
from queue import Queue
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QLineEdit,
    QFrame, QGraphicsDropShadowEffect, QDialog, QMessageBox, QFormLayout
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal

# Modüllerimiz
from src.inference import SignLanguagePredictor
from src.avatar_module import AvatarRenderer, AvatarPlayer
from src.tcp_server import TcpBroadcastServer
from src.keypoints_utils import extract_raw_landmarks
from src import db

class RegisterWindow(QDialog):
    """
    Kullanıcı kayıt penceresini temsil eden arayüz sınıfı.
    Kullanıcı adı ve şifre bilgilerini alarak veritabanına yeni bir kullanıcı eklenmesini sağlar.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kayıt Ol")
        self.setFixedSize(300, 200)
        self.setStyleSheet("""
            QDialog { background-color: #1e1e2e; color: #cdd6f4; }
            QLabel { color: #cdd6f4; font-weight: bold; }
            QLineEdit { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 5px; padding: 5px; }
            QPushButton { background-color: #cba6f7; color: #11111b; font-weight: bold; border-radius: 5px; padding: 8px; }
            QPushButton:hover { background-color: #b4befe; }
        """)
        
        layout = QFormLayout()
        
        self.username_input = QLineEdit()
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        
        layout.addRow(QLabel("Kullanıcı Adı:"), self.username_input)
        layout.addRow(QLabel("Şifre:"), self.password_input)
        
        self.btn_register = QPushButton("Kayıt Ol")
        self.btn_register.clicked.connect(self.handle_register)
        layout.addWidget(self.btn_register)
        
        self.setLayout(layout)
        
    def handle_register(self):
        """
        'Kayıt Ol' butonuna tıklandığında çalışır.
        Arayüzden alınan kullanıcı adı ve şifreyi db modülü üzerinden veritabanına kaydeder.
        İşlem sonucuna göre (başarılı veya hata) ekranda bir mesaj kutusu gösterir.
        """
        username = self.username_input.text()
        password = self.password_input.text()
        
        success, message = db.register_user(username, password)
        if success:
            QMessageBox.information(self, "Başarılı", message)
            self.accept()
        else:
            QMessageBox.warning(self, "Hata", message)

class LoginWindow(QDialog):
    """
    Kullanıcı giriş penceresini temsil eden arayüz sınıfı.
    Kullanıcı adı ve şifre doğrulaması yaparak sisteme giriş işlemlerini yönetir.
    Giriş başarılı olduğunda ana pencerenin (MainWindow) açılmasını tetikler.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Giriş Yap")
        self.setFixedSize(300, 250)
        self.user_id = None
        self.setStyleSheet("""
            QDialog { background-color: #1e1e2e; color: #cdd6f4; }
            QLabel { color: #cdd6f4; font-weight: bold; }
            QLineEdit { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 5px; padding: 5px; }
            QPushButton { background-color: #89b4fa; color: #11111b; font-weight: bold; border-radius: 5px; padding: 8px; }
            QPushButton:hover { background-color: #74c7ec; }
            QPushButton#BtnReg { background-color: #313244; color: #cdd6f4; }
            QPushButton#BtnReg:hover { background-color: #45475a; }
        """)
        
        layout = QVBoxLayout()
        
        form_layout = QFormLayout()
        self.username_input = QLineEdit()
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        
        form_layout.addRow(QLabel("Kullanıcı Adı:"), self.username_input)
        form_layout.addRow(QLabel("Şifre:"), self.password_input)
        
        layout.addLayout(form_layout)
        
        self.btn_login = QPushButton("Giriş Yap")
        self.btn_login.clicked.connect(self.handle_login)
        layout.addWidget(self.btn_login)
        
        self.btn_register = QPushButton("Yeni Kayıt Ol")
        self.btn_register.setObjectName("BtnReg")
        self.btn_register.clicked.connect(self.open_register)
        layout.addWidget(self.btn_register)
        
        self.setLayout(layout)
        
    def handle_login(self):
        """
        'Giriş Yap' butonuna basıldığında tetiklenir.
        Veritabanından kullanıcı bilgilerini doğrular. Eğer giriş başarılıysa,
        ilgili kullanıcının ID'sini (user_id) saklar ve pencereyi onaylayarak kapatır.
        """
        username = self.username_input.text()
        password = self.password_input.text()
        
        success, user_id = db.authenticate_user(username, password)
        if success:
            self.user_id = user_id
            QMessageBox.information(self, "Başarılı", "Giriş başarılı!")
            self.accept()
        else:
            QMessageBox.warning(self, "Hata", "Kullanıcı adı veya şifre hatalı!")
            
    def open_register(self):
        reg_win = RegisterWindow()
        reg_win.exec_()
class CameraReader(QThread):
    """
    Sadece I/O (Girdi/Çıktı) işlemlerinden sorumlu arka plan iş parçacığı (Thread).
    Kameradan sürekli olarak kare (frame) okuyup bunları bir kuyruğa (Queue) atar.
    Bu sayede kameradan görüntü alma işlemi, ağır olan tahmin (MediaPipe/LSTM) işlemlerini beklemez,
    uygulamanın donması (lag) engellenir.
    """

    def __init__(self, frame_queue: Queue):
        super().__init__()
        self._run_flag = True
        self._queue = frame_queue

    def run(self):
        cap = cv2.VideoCapture(0)

        # 720p — MediaPipe zaten dahili olarak küçültür, 1080p israf
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                # maxsize=1: Eski kareyi at, her zaman en taze kareyi tut
                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                    except Exception:
                        pass
                self._queue.put(frame)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


class VideoThread(QThread):
    """
    Görüntü İşleme ve Tahmin iş parçacığı (Worker Thread).
    CameraReader'ın kuyruğa attığı taze kareleri alır, SignLanguagePredictor kullanarak
    MediaPipe ile koordinat çıkarımı ve LSTM modeli ile işaret dili tahmini yapar.
    Sonuçları (görüntü, tahmin metni, ve ham landmark verilerini) sinyaller aracılığıyla ana iş parçacığına (UI) iletir.
    """
    change_pixmap_signal = pyqtSignal(np.ndarray)
    prediction_signal = pyqtSignal(str)
    landmarks_signal = pyqtSignal(dict)

    def __init__(self, frame_queue: Queue):
        super().__init__()
        self._run_flag = True
        self._queue = frame_queue
        self.predictor = SignLanguagePredictor()
        self.prediction_enabled = True

    def run(self):
        while self._run_flag:
            # Kuyruktan en taze kareyi al (kamera okumayı beklemeden)
            try:
                frame = self._queue.get(timeout=0.1)
            except Exception:
                continue

            image, results, action, prob = self.predictor.predict(frame, enable_prediction=self.prediction_enabled)

            if action:
                self.prediction_signal.emit(f"{action} (%{int(prob * 100)})")

            # Her kare: ham landmark verisi Unity'ye gönderilecek
            raw_lm = extract_raw_landmarks(results)
            self.landmarks_signal.emit(raw_lm)

            self.change_pixmap_signal.emit(image)

    def stop(self):
        self._run_flag = False
        self.wait()


class UdpVideoReceiver(QThread):
    """
    Unity tarafından UDP üzerinden yayınlanan (stream) canlı avatar görüntülerini almak için kullanılan dinleyici (Thread).
    QPixmap arka planda oluşturulamadığı için, gelen byte verisi OpenCV ile bir NumPy dizisine (RGB formatında) 
    dönüştürülür ve sinyal olarak ana ekrana (UI thread) iletilerek orada gösterilmesi sağlanır.
    """
    # Raw numpy array gonderip main thread'de QPixmap'e ceviriyoruz.
    frame_data_signal = pyqtSignal(np.ndarray)

    def __init__(self, port=5556):
        super().__init__()
        self.port = port
        self._run_flag = True
        self._frame_count = 0
        self.sock = None

    def run(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536 * 10)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind(("127.0.0.1", self.port))
            self.sock.settimeout(1.0)
            print(f"[UdpReceiver] Port {self.port} dinleniyor...")
        except Exception as e:
            print(f"[UdpReceiver] HATA: Socket baglanilamadi: {e}")
            return

        while self._run_flag:
            try:
                data, addr = self.sock.recvfrom(65507)
                if data:
                    np_arr = np.frombuffer(data, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.frame_data_signal.emit(rgb_image)
                        self._frame_count += 1
                        if self._frame_count == 1:
                            print(f"[UdpReceiver] Ilk frame alindi! ({len(data)} byte)")
                        elif self._frame_count % 300 == 0:
                            print(f"[UdpReceiver] {self._frame_count} frame alindi")
                    else:
                        print(f"[UdpReceiver] UYARI: JPEG decode basarisiz ({len(data)} byte)")
            except socket.timeout:
                pass
            except Exception as e:
                if self._run_flag:
                    print(f"[UdpReceiver] Hata: {e}")

    def stop(self):
        self._run_flag = False
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
        self.wait()



class MainWindow(QMainWindow):
    """
    Uygulamanın ana arayüz (GUI) sınıfı.
    Sol panelde kameradan alınan canlı görüntüyü ve tahmin edilen işaret dilini gösterir.
    Sağ panelde ise metinden işaret diline çeviriyi Unity Avatar (UDP/TCP) veya yedek MP4 videolar üzerinden sağlar.
    Aktif Öğrenme (Kullanıcı geribildirimi) sistemini ve çok parçacıklı (Thread) işlemleri koordine eder.
    """
    def __init__(self, user_id=None):
        super().__init__()
        self.user_id = user_id
        self.setWindowTitle("AI-Sign: İşaret Dili Çeviricisi")
        self.setGeometry(100, 100, 1200, 700)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e2e;
                color: #cdd6f4;
            }
            QFrame#PanelFrame {
                background-color: #27293d;
                border-radius: 12px;
            }
            QLabel {
                color: #cdd6f4;
            }
            QLabel#TitleLabel {
                font-family: 'Segoe UI', 'Arial';
                font-size: 20px;
                font-weight: bold;
                color: #cdd6f4;
                margin-bottom: 10px;
            }
            QLabel#VideoLabel {
                border: 2px solid #313244;
                border-radius: 8px;
                background-color: #11111b;
            }
            QLabel#SuggestionLabel {
                font-family: 'Segoe UI', 'Arial';
                font-size: 18px;
                font-weight: bold;
                color: #f9e2af;
                background-color: #313244;
                border-radius: 8px;
                padding: 10px;
                margin-top: 10px;
                margin-bottom: 10px;
            }
            QPushButton {
                font-family: 'Segoe UI', 'Arial';
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
                padding: 10px 15px;
                color: white;
            }
            QPushButton#BtnConfirm {
                background-color: #a6e3a1;
                color: #11111b;
            }
            QPushButton#BtnConfirm:hover {
                background-color: #94e2d5;
            }
            QPushButton#BtnConfirm:pressed {
                background-color: #89b4fa;
            }
            QPushButton#BtnReject {
                background-color: #f38ba8;
                color: #11111b;
            }
            QPushButton#BtnReject:hover {
                background-color: #eba0ac;
            }
            QPushButton#BtnReject:pressed {
                background-color: #f2cdcd;
            }
            QPushButton#BtnTogglePred {
                background-color: #fab387;
                color: #11111b;
                margin-top: 10px;
            }
            QPushButton#BtnTogglePred:checked {
                background-color: #89b4fa;
            }
            QPushButton#BtnTranslate {
                background-color: #cba6f7;
                color: #11111b;
                margin-top: 10px;
            }
            QPushButton#BtnTranslate:hover {
                background-color: #b4befe;
            }
            QPushButton#BtnTranslate:pressed {
                background-color: #89b4fa;
            }
            QPushButton#BtnToggleMode {
                background-color: #89b4fa;
                color: #11111b;
                margin-top: 5px;
            }
            QPushButton#BtnToggleMode:checked {
                background-color: #f38ba8;
            }
            QLineEdit {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 8px;
                padding: 10px;
                font-family: 'Segoe UI', 'Arial';
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 1px solid #89b4fa;
            }
            QLabel#HistoryLabel {
                color: #a6adc8;
                padding: 10px;
                border-top: 1px solid #45475a;
                margin-top: 15px;
            }
            QLabel#AvatarLabel {
                border: 2px solid #cba6f7;
                border-radius: 8px;
                background-color: #11111b;
                color: #cba6f7;
                font-size: 14px;
            }
            QLabel#TcpStatus {
                padding: 5px;
                margin-top: 5px;
                margin-bottom: 5px;
            }
        """)

        self.avatar_renderer = AvatarRenderer()

        # Avatar video oynatma — sinyal tabanlı AvatarPlayer
        self.avatar_player = AvatarPlayer()
        self.avatar_player.frame_ready.connect(self._on_avatar_frame)
        self.avatar_player.playback_finished.connect(self._on_avatar_finished)

        # TCP Sunucu — Unity'ye tahmin göndermek için
        self.tcp_server = TcpBroadcastServer(host="127.0.0.1", port=5555)
        self.tcp_server.start()

        # UDP Sunucu — Unity'den Canlı Video almak için
        self.udp_receiver = UdpVideoReceiver(port=5556)
        self.udp_receiver.frame_data_signal.connect(self._on_unity_video_frame)
        self.udp_receiver.start()

        self.init_ui()

    def create_shadow(self):
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 4)
        return shadow

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # --- SOL PANEL: Sign-to-Text (Kamera) ---
        left_frame = QFrame()
        left_frame.setObjectName("PanelFrame")
        left_frame.setGraphicsEffect(self.create_shadow())
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(10)

        label_cam_title = QLabel("İşaret -> Metin (Kamera)")
        label_cam_title.setObjectName("TitleLabel")
        left_layout.addWidget(label_cam_title)

        self.image_label = QLabel(self)
        self.image_label.setObjectName("VideoLabel")
        self.image_label.setFixedSize(640, 480)
        self.image_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.image_label)

        # ÖNERİ KUTUSU
        self.suggestion_label = QLabel("Öneri: ...")
        self.suggestion_label.setObjectName("SuggestionLabel")
        left_layout.addWidget(self.suggestion_label)

        # ONAY BUTONLARI
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)
        self.btn_confirm = QPushButton("✅ Onayla (Evet)")
        self.btn_confirm.setObjectName("BtnConfirm")
        self.btn_confirm.setCursor(Qt.PointingHandCursor)
        self.btn_confirm.clicked.connect(self.confirm_suggestion)

        self.btn_reject = QPushButton("❌ Reddet (Hayır)")
        self.btn_reject.setObjectName("BtnReject")
        self.btn_reject.setCursor(Qt.PointingHandCursor)
        self.btn_reject.clicked.connect(self.reject_suggestion)

        btn_layout.addWidget(self.btn_confirm)
        btn_layout.addWidget(self.btn_reject)
        left_layout.addLayout(btn_layout)

        self.btn_toggle_pred = QPushButton("🛑 Tahmini Durdur")
        self.btn_toggle_pred.setObjectName("BtnTogglePred")
        self.btn_toggle_pred.setCursor(Qt.PointingHandCursor)
        self.btn_toggle_pred.setCheckable(True)
        self.btn_toggle_pred.clicked.connect(self.toggle_prediction)
        left_layout.addWidget(self.btn_toggle_pred)

        # GEÇMİŞ
        self.text_output = QLabel("Konuşma Geçmişi: ...")
        self.text_output.setObjectName("HistoryLabel")
        self.text_output.setWordWrap(True)
        left_layout.addWidget(self.text_output)

        left_layout.addStretch()
        main_layout.addWidget(left_frame)

        # --- SAĞ PANEL: Text-to-Sign (Avatar) ---
        right_frame = QFrame()
        right_frame.setObjectName("PanelFrame")
        right_frame.setGraphicsEffect(self.create_shadow())
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(10)

        label_avatar_title = QLabel("Metin -> İşaret (Avatar)")
        label_avatar_title.setObjectName("TitleLabel")
        right_layout.addWidget(label_avatar_title)

        self.avatar_label = QLabel(self)
        self.avatar_label.setObjectName("AvatarLabel")
        self.avatar_label.setFixedSize(400, 400)
        self.avatar_label.setAlignment(Qt.AlignCenter)
        self.avatar_label.setText(
            "\n🎮 Unity Avatar Bekleniyor...\n\n"
            "Unity'yi başlatıp Play'e basın.\n"
            "Canlı avatar burada görünecek.\n\n"
            "TCP: localhost:5555\n"
            "UDP: localhost:5556"
        )
        self._unity_streaming = False  # Unity'den canlı frame geldi mi?
        right_layout.addWidget(self.avatar_label)

        # TCP Bağlantı Durumu
        self.tcp_status_label = QLabel("🔴 Unity bağlı değil")
        self.tcp_status_label.setObjectName("TcpStatus")
        self.tcp_status_label.setStyleSheet("color: #f38ba8;")
        right_layout.addWidget(self.tcp_status_label)

        # Bağlantı durumu güncelleme zamanlayıcı
        self.tcp_status_timer = QTimer()
        self.tcp_status_timer.timeout.connect(self._update_tcp_status)
        self.tcp_status_timer.start(1000)  # Her saniye kontrol et

        # Metin Girişi
        self.input_text = QLineEdit()
        self.input_text.setPlaceholderText("Çevrilecek metni yazın...")
        right_layout.addWidget(self.input_text)

        btn_translate = QPushButton("Çevir ve Oynat")
        btn_translate.setObjectName("BtnTranslate")
        btn_translate.setCursor(Qt.PointingHandCursor)
        btn_translate.clicked.connect(self.play_avatar)
        right_layout.addWidget(btn_translate)

        self.btn_toggle_mode = QPushButton("Mod: Avatar")
        self.btn_toggle_mode.setObjectName("BtnToggleMode")
        self.btn_toggle_mode.setCheckable(True)
        self.btn_toggle_mode.setCursor(Qt.PointingHandCursor)
        self.btn_toggle_mode.clicked.connect(self.toggle_avatar_video_mode)
        self.force_video_mode = False
        right_layout.addWidget(self.btn_toggle_mode)

        right_layout.addStretch()
        main_layout.addWidget(right_frame)

        main_widget.setLayout(main_layout)

        # --- 2 Aşamalı Üretici-Tüketici Mimarisi ---
        # CameraReader (I/O) → Queue(maxsize=1) → VideoThread (İşleme)
        self._frame_queue = Queue(maxsize=1)

        self.camera_reader = CameraReader(self._frame_queue)
        self.camera_reader.start()

        self.thread = VideoThread(self._frame_queue)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.prediction_signal.connect(self.update_suggestion)
        self.thread.landmarks_signal.connect(self._on_landmarks)
        self.thread.start()

        self.current_suggestion = ""
        self.conversation_history = []

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def update_suggestion(self, text):
        if text != self.current_suggestion:
            self.current_suggestion = text
            self.suggestion_label.setText(f"Öneri: {text} ?")

            # Unity'ye tahmin gönder (her yeni öneri)
            word = text.split(" ")[0]
            try:
                conf_str = text.split("%")[1].rstrip(")")
                confidence = int(conf_str) / 100.0
            except (IndexError, ValueError):
                confidence = 0.0
            self.tcp_server.broadcast({
                "type": "P",
                "word": word,
                "confidence": confidence,
                "timestamp": time.time(),
            })

            # Tahmini otomatik duraklat (Auto-Pause) ki kullanıcı onaylayana kadar yeni tahmin yapmasın
            if not self.btn_toggle_pred.isChecked():
                # En son tahmin edilen hareketi yedekle (tahmin durdurulup sıfırlanmadan önce)
                self.last_predicted_sequence = list(self.thread.predictor.sequence)
                
                self.btn_toggle_pred.setChecked(True)
                self.toggle_prediction()

    def confirm_suggestion(self):
        """
        Kullanıcı önerilen işareti doğru bulduğunda ('Onayla' butonu) tetiklenir.
        Doğru bilinen kelimenin verisini Aktif Öğrenme (Active Learning) için kaydeder,
        konuşma geçmişine ekler ve sağdaki Avatar'ın bu kelimeyi canlandırması için TCP sinyali gönderir.
        """
        if self.current_suggestion:
            word_cap = self.current_suggestion.split(" ")[0] # Baş harfi büyük, örn: "Oda"
            
            # DB LOG
            if self.user_id:
                db.log_feedback(self.user_id, word_cap, True)
            
            # Arka planda Aktif Öğrenme için onaylanan kare sekansını kaydet
            self.save_feedback_data(word_cap)
            
            # Başarılı tahminden sonra kara listeyi temizle
            self.thread.predictor.clear_blacklist()
            
            word = word_cap.lower()
            self.conversation_history.append(word)
            self.text_output.setText("Konuşma: " + " ".join(self.conversation_history))
            
            # Unity'e Pürüzsüz Referans Animasyonu Gönder (Playback Mode)
            self.tcp_server.stream_reference_animation(word)
            
            self.current_suggestion = ""
            self.suggestion_label.setText("Öneri: ...")
            
            # Tahmini otomatik olarak devam ettir (unpause)
            if self.btn_toggle_pred.isChecked():
                self.btn_toggle_pred.setChecked(False)
                self.toggle_prediction()

    def reject_suggestion(self):
        """
        Kullanıcı önerilen işareti yanlış bulduğunda ('Reddet' butonu) tetiklenir.
        İlgili kelimeyi modelin tahmin kara listesine (blacklist) ekler, böylece 
        model aynı yanlış kelimeyi üst üste tahmin etmekten kaçınır.
        """
        if self.current_suggestion:
            word_cap = self.current_suggestion.split(" ")[0] # Baş harfi büyük, örn: "Oda"
            
            # DB LOG
            if self.user_id:
                db.log_feedback(self.user_id, word_cap, False)
            
            # Kelimeyi geçici olarak engellenenler listesine ekle
            self.thread.predictor.blacklist_word(word_cap)
            
            self.current_suggestion = ""
            self.suggestion_label.setText("Öneri: ...")
            
            # Tahmini otomatik olarak devam ettir (unpause)
            if self.btn_toggle_pred.isChecked():
                self.btn_toggle_pred.setChecked(False)
                self.toggle_prediction()

    def save_feedback_data(self, word):
        """Kullanıcının doğru olarak onayladığı 16 karelik hareket sekansını
        data/WORD/ klasörüne yeni bir sequence olarak kaydeder (Aktif Öğrenme)."""
        try:
            from pathlib import Path
            import os
            
            # Yedeklenmiş sekansı al
            sequence = getattr(self, 'last_predicted_sequence', [])
            
            if len(sequence) != 16:
                print(f"[ActiveLearning] UYARI: Kare sekansı uzunluğu 16 değil ({len(sequence)}), kaydedilmedi.")
                return
                
            # data/KELIME klasörünü bul veya oluştur
            root_dir = Path(__file__).resolve().parent
            word_dir = str(root_dir / "data" / word)
            if not os.path.exists(word_dir):
                os.makedirs(word_dir)
                
            # Bir sonraki boş sequence klasör numarasını bul
            existing_seqs = [d for d in os.listdir(word_dir) if os.path.isdir(os.path.join(word_dir, d))]
            seq_nums = []
            for s in existing_seqs:
                try:
                    seq_nums.append(int(s))
                except ValueError:
                    pass
            next_seq_num = max(seq_nums) + 1 if seq_nums else 0
            
            # Yeni sequence klasörünü oluştur
            new_seq_dir = os.path.join(word_dir, str(next_seq_num))
            os.makedirs(new_seq_dir)
            
            # 16 kareyi .npy olarak kaydet
            for i, frame_data in enumerate(sequence):
                npy_path = os.path.join(new_seq_dir, f"{i}.npy")
                np.save(npy_path, frame_data)
                
            print(f"[ActiveLearning] BAŞARILI: '{word}' için yeni hareket verisi kaydedildi (Seq: {next_seq_num})!")
            self.last_predicted_sequence = [] # Kayıttan sonra yedek sekansı temizle
        except Exception as e:
            print(f"[ActiveLearning] HATA: Veri kaydedilemedi: {e}")

    def toggle_prediction(self):
        """
        Tahmini Durdur / Başlat butonunun mantığını yönetir.
        Tıklandığında VideoThread içindeki prediction_enabled bayrağını (flag) değiştirerek
        MediaPipe/LSTM işlemlerini duraklatır veya sürdürür.
        """
        if self.btn_toggle_pred.isChecked():
            self.thread.prediction_enabled = False
            self.btn_toggle_pred.setText("▶️ Tahmini Başlat")
            if self.current_suggestion:
                self.suggestion_label.setText(f"Öneri: {self.current_suggestion} ? [DURAKLATILDI]")
            else:
                self.suggestion_label.setText("Öneri: [DURDURULDU]")
        else:
            self.thread.prediction_enabled = True
            self.btn_toggle_pred.setText("🛑 Tahmini Durdur")
            if self.current_suggestion:
                self.suggestion_label.setText(f"Öneri: {self.current_suggestion} ?")
            else:
                self.suggestion_label.setText("Öneri: ...")

    def toggle_avatar_video_mode(self):
        """
        'Mod: Avatar' ve 'Mod: Video' butonu tıklandığında çalışır.
        Kullanıcının Unity Avatar'ı yerine gerçek insan (MP4) videosunu kullanmak istemesi 
        durumunda devreye girer. Unity'den gelen görüntüleri engeller.
        """
        if self.btn_toggle_mode.isChecked():
            self.force_video_mode = True
            self.btn_toggle_mode.setText("Mod: Video")
            # Avatar oynatıcıyı durdur, eğer Unity'den frame geliyorsa ekrandan temizlemek için
            self.avatar_player.stop()
            self.avatar_label.setText("Mod: Video\nLütfen kelime girip 'Çevir ve Oynat'a basın.")
        else:
            self.force_video_mode = False
            self.btn_toggle_mode.setText("Mod: Avatar")
            if self._unity_streaming:
                self.avatar_label.setText("Avatar Yayını Sürdürülüyor...")
            else:
                self.avatar_label.setText("Avatar Bekleniyor...")

    def _on_landmarks(self, raw_lm):
        """Her karedeki ham landmark verisini TCP ile Unity'ye gönderir.
        Kullanıcı isteği: Avatar kamerayı taklit etmesin, yalnızca metne hareket etsin.
        """
        pass
        # self.tcp_server.broadcast(raw_lm)

    def _update_tcp_status(self):
        """TCP bağlantı durumunu UI'da gösterir."""
        count = self.tcp_server.client_count
        if count > 0:
            self.tcp_status_label.setText(f"🟢 Unity bağlı ({count} istemci)")
            self.tcp_status_label.setStyleSheet("color: #a6e3a1;")
        else:
            self.tcp_status_label.setText("🔴 Unity bağlı değil")
            self.tcp_status_label.setStyleSheet("color: #f38ba8;")

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    # --- Avatar Video Oynatma (AvatarPlayer sinyal tabanlı — UI thread'i bloklamaz) ---

    def play_avatar(self):
        """
        'Çevir ve Oynat' butonuna basıldığında tetiklenir.
        Eğer Unity Avatar bağlıysa ve aktifse, TCP üzerinden kelime gönderilir.
        Eğer Video Modu aktifse veya Unity bağlı değilse (fallback), kelimeye ait
        MP4 formatındaki video bulunur ve sağ panelde oynatılır.
        """
        text = self.input_text.text().strip().lower()
        if not text:
            return

        force_video = getattr(self, 'force_video_mode', False)

        if not force_video:
            # Unity Avatarı için TCP'den referans akışı başlat
            self.tcp_server.stream_reference_animation(text)

        # Fallback: Unity bağlı değilse veya force_video aktifse → MP4 video oynat
        if not self._unity_streaming or force_video:
            video_path = self.avatar_renderer.get_video_path(text)
            if video_path:
                self.avatar_player.play_path(video_path)
            else:
                self.avatar_label.setText(f"'{text}' için video bulunamadı.")

    def _on_avatar_frame(self, pixmap):
        """AvatarPlayer'dan gelen her kareyi avatar_label'a yansıtır."""
        scaled = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.avatar_label.setPixmap(scaled)

    def _on_avatar_finished(self):
        """Video bittiğinde kullanıcıya bilgi verir."""
        self.avatar_label.setText("Oynatma Bitti.")

    def _on_unity_video_frame(self, rgb_image):
        """Unity'den UDP üzerinden gelen canlı avatar frame'ini avatar_label'a basar.
        rgb_image: np.ndarray (RGB format, main thread'de QPixmap'e çevrilir)"""
        if getattr(self, 'force_video_mode', False):
            return
            
        if not self._unity_streaming:
            self._unity_streaming = True
            self.avatar_player.stop()
            print("[MainWindow] Unity stream basladi!")
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.avatar_label.setPixmap(scaled)

    def closeEvent(self, event):
        self.tcp_server.stop()
        self.avatar_player.stop()
        self.camera_reader.stop()
        self.thread.stop()
        self.udp_receiver.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    db.init_db()
    
    login = LoginWindow()
    if login.exec_() == QDialog.Accepted:
        window = MainWindow(user_id=login.user_id)
        window.show()
        sys.exit(app.exec_())
    else:
        sys.exit(0)
