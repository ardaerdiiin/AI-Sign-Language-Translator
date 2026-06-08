<div align="center">
  <h1>🤖 AI-Sign: İşaret Dili Çeviricisi</h1>
  <p>
    <b>Türk İşaret Dili (TİD) ile yazılı metin arasında gerçek zamanlı, çift yönlü çeviri yapan yapay zeka destekli masaüstü uygulaması.</b>
  </p>
  
  [![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)](https://www.python.org/)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-TFLite-FF6F00?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/)
  [![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic-00A67E?style=for-the-badge)](https://mediapipe.dev/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

  <br />
</div>





## 🚀 Proje Hakkında

**AI-Sign**, işitme engelli bireyler ile işaret dili bilmeyen kişiler arasındaki iletişim engelini kaldırmayı amaçlayan uçtan uca (end-to-end) bir yazılımdır. Standart bir web kamerası aracılığıyla işaret dilini algılar, yapay zeka ağı (Transformer Encoder) üzerinden tahminler yapar ve metne döker. Aynı zamanda metin olarak yazılan kelimeleri, asenkron oynatıcı modülü ve (opsiyonel) 3D Avatar desteği ile işaret diline çevirir.

### 🌟 Temel Özellikler
- **🎥 İşaret → Metin (Sign-to-Text):** Gerçek zamanlı olarak el, vücut ve duruş takibi yaparak kelimeleri metne çevirme.
- **⌨️ Metin → İşaret (Text-to-Sign):** Metin girişini işaret dili video sekanslarına veya Avatar hareketlerine dönüştürme.
- **⚡ Yüksek Performans:** TFLite entegrasyonu ile tahmin süreleri ~3ms.
- **🛡️ Stabilite Filtresi:** Hatalı veya rastgele hareketleri filtreleyen "Arka arkaya yüksek güven" mekanizması.
- **🧵 Multi-threading Yapısı:** Görüntü işleme ve yapay zeka çıkarımlarının (inference) arayüzü kitlememesi için "Producer-Consumer" yapısı kullanılmıştır.

---

## 🛠️ Kullanılan Teknolojiler

- **Python** - Ana dil
- **TensorFlow / Keras / TFLite** - Transformer Encoder (Yapay Zeka Modeli)
- **MediaPipe** - İskelet ve Eklem Tespiti (Holistic/Pose/Hands)
- **OpenCV** - Kamera verisi işleme
- **PyQt5** - Asenkron Masaüstü Arayüzü (GUI)
- **NumPy** - 194 boyutlu özellik mühendisliği (Feature Engineering) hesaplamaları

---

## ⚙️ Kurulum

Bilgisayarınızda Python kurulu olmalıdır.

1. Repoyu bilgisayarınıza klonlayın:
```bash
git clone https://github.com/ardaerdiiin/AI-Sign-Language-Translator.git
cd AI-Sign-Language-Translator
```

2. Sanal ortam (Virtual Environment) oluşturun ve aktif edin (Önerilir):
```bash
python -m venv .venv
# Windows için:
.venv\Scripts\activate
# Mac/Linux için:
source .venv/bin/activate
```

3. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

---

## 💻 Kullanım

Uygulamayı başlatmak için ana dizinde şu komutu çalıştırın:

```bash
python main.py
```
- **Sol Panel:** Kameranız açılacak ve sistem hareketlerinizi algılamaya başlayacaktır. Bir kelime algılandığında ekranda beliren "Öneri" kısmından onaylayıp/reddederek sohbet penceresini doldurabilirsiniz.
- **Sağ Panel:** Alt kısımdaki metin kutusuna kelime yazarak uygulamanın bu kelimeyi işaret diline çevirmesini (görselleştirmesini) sağlayabilirsiniz.

---

## 🤝 Katkıda Bulunma (Contributing)
Bu projeyi geliştirmek isterseniz katkılarınızı büyük bir memnuniyetle kabul ediyoruz! Yeni kelimeler eklemek veya koda katkıda bulunmak için lütfen [CONTRIBUTING.md](CONTRIBUTING.md) dosyamıza göz atın.

1. Bu repoyu Fork'layın
2. Yeni bir dal (branch) oluşturun (`git checkout -b feature/YeniOzellik`)
3. Değişikliklerinizi commit'leyin (`git commit -m 'Harika bir özellik eklendi'`)
4. Dalınızı push'layın (`git push origin feature/YeniOzellik`)
5. Bir Pull Request açın!

---

## 📄 Lisans
Bu proje **MIT Lisansı** altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakabilirsiniz.
