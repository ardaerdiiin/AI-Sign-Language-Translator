"""
TCP Broadcast Server — Python → Unity köprüsü.

localhost:5555 üzerinde dinler, bağlı tüm istemcilere JSON satırları gönderir.
Unity tarafındaki PythonBridge.cs bu sunucuya TcpClient ile bağlanır.
"""

import json
import socket
import threading
import time


class TcpBroadcastServer(threading.Thread):
    """
    Arka planda (Thread) sürekli çalışan TCP sunucusu sınıfı.
    Unity (C#) ile Python yapay zeka bölümü arasındaki köprü iletişimi sağlar.
    Birden fazla istemciyi (client) aynı anda kabul edebilir ve broadcast() metoduyla 
    bağlı olan tüm istemcilere (örneğin Unity Avatarına) JSON formatında
    iskelet koordinatları (landmark) veya komut (kelime) verisi gönderir.
    """

    def __init__(self, host="127.0.0.1", port=5555):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self._clients = []          # Bağlı istemci soketleri
        self._clients_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._server_socket = None
        
        # Oynatma (Playback) Kontrolleri
        self._is_playing_ref = False
        self._playback_thread = None

    @property
    def client_count(self):
        """Bağlı istemci sayısını döndürür (thread-safe)."""
        with self._clients_lock:
            return len(self._clients)

    def run(self):
        """Sunucu döngüsü — yeni bağlantıları kabul eder."""
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.settimeout(1.0)  # accept() timeout — stop kontrolü için

        try:
            self._server_socket.bind((self.host, self.port))
            self._server_socket.listen(5)
            print(f"[TcpServer] Dinleniyor: {self.host}:{self.port}")
        except OSError as e:
            print(f"[TcpServer] Bağlanamadı: {e}")
            return

        while not self._stop_event.is_set():
            try:
                client_socket, addr = self._server_socket.accept()
                with self._clients_lock:
                    self._clients.append(client_socket)
                print(f"[TcpServer] Yeni bağlantı: {addr}")
            except socket.timeout:
                continue
            except OSError:
                break

        self._cleanup()

    def broadcast(self, data: dict):
        """
        Bağlı tüm istemcilere (Unity) veri sözlüğünü (dict) JSON satırına çevirerek yollar.
        Eğer sistem bir kelimenin referans animasyonunu oynatıyorsa (playback), 
        kameradan gelen canlı (type='L') iskelet verilerini kasıtlı olarak engeller.
        Böylece avatar hareketleri (canlı vs kayıtlı) birbirine karışmaz.
        """
        if self._is_playing_ref and data.get("type") == "L":
            return
            
        self._do_broadcast(data)

    def _do_broadcast(self, data: dict):
        """Asıl gönderim fonksiyonu (Playback ve Broadcast tarafından kullanılır)."""
        message = json.dumps(data, ensure_ascii=False) + "\n"
        encoded = message.encode("utf-8")

        dead_clients = []

        with self._clients_lock:
            for client in self._clients:
                try:
                    client.sendall(encoded)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    dead_clients.append(client)

            for client in dead_clients:
                self._clients.remove(client)
                try:
                    client.close()
                except OSError:
                    pass

        if dead_clients:
            print(f"[TcpServer] {len(dead_clients)} istemci bağlantısı temizlendi.")

    def stop(self):
        """Sunucuyu ve tüm bağlantıları temiz şekilde kapatır."""
        print("[TcpServer] Kapatılıyor...")
        self._stop_event.set()

        if self._server_socket:
            try:
                self._server_socket.close()
            except OSError:
                pass

        self.join(timeout=3)

    def stream_reference_animation(self, word: str):
        """
        'Çevir ve Oynat' özelliğinde (Avatar modunda) çağrılır.
        Unity Avatarının kelimeyi canlandırabilmesi için, önceden 'data/ref_landmarks' içine
        kaydedilmiş olan JSON dosyasını (hareket dizisini) okur ve 30 FPS (saniyede 30 kare) 
        hızında senkronize şekilde Unity'e gönderir. Oynatma süresince canlı kamera verisi yoksayılır.
        """
        if self._is_playing_ref:
            return  # Zaten bir şey oynatılıyor

        def playback():
            self._is_playing_ref = True
            try:
                from pathlib import Path
                # Proje kökünden data/ref_landmarks yolunu bul
                ref_path = Path(__file__).resolve().parent.parent / "data" / "ref_landmarks" / f"{word}.json"
                
                if not ref_path.exists():
                    print(f"[TcpServer] Uyarı: {word} için referans animasyon bulunamadı ({ref_path}).")
                    return
                
                with open(ref_path, "r", encoding="utf-8") as f:
                    frames = json.load(f)
                
                print(f"[TcpServer] Playback başladı: {word} ({len(frames)} kare)")
                
                fps = 30.0
                frame_duration = 1.0 / fps
                start_time = time.time()
                
                for i, frame in enumerate(frames):
                    if self._stop_event.is_set():
                        break
                    # P bypassı için _do_broadcast kullanıyoruz
                    self._do_broadcast(frame)
                    
                    # Drift hesaplaması ile dinamik uyku: hedef zaman hesaplanıp uyutulur
                    expected_time = start_time + (i + 1) * frame_duration
                    sleep_time = expected_time - time.time()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    
            except Exception as e:
                print(f"[TcpServer] Playback hatası: {e}")
            finally:
                self._is_playing_ref = False
                print(f"[TcpServer] Playback bitti: {word}")

        self._playback_thread = threading.Thread(target=playback, daemon=True)
        self._playback_thread.start()

    def _cleanup(self):
        """Tüm istemci bağlantılarını kapatır."""
        with self._clients_lock:
            for client in self._clients:
                try:
                    client.close()
                except OSError:
                    pass
            self._clients.clear()

        if self._server_socket:
            try:
                self._server_socket.close()
            except OSError:
                pass

        print("[TcpServer] Kapatıldı.")


# ------------------------------------------------------------------
# Hızlı test: python -m src.tcp_server
# ------------------------------------------------------------------
if __name__ == "__main__":
    server = TcpBroadcastServer()
    server.start()

    try:
        i = 0
        while True:
            server.broadcast({"word": "Test", "confidence": 0.99, "timestamp": time.time()})
            print(f"[Test] Mesaj gönderildi #{i}")
            i += 1
            time.sleep(2)
    except KeyboardInterrupt:
        server.stop()
