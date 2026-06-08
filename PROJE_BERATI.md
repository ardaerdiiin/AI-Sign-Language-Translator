# PROJE BERATI — AI-Sign: İşaret Dili Çeviricisi

---

## 1. Proje Kimliği

| Bilgi | Detay |
|---|---|
| **Proje Adı** | AI-Sign: İşaret Dili Çeviricisi |
| **Proje Türü** | Yapay Zeka Tabanlı Masaüstü Uygulaması |
| **Platform** | Windows (Masaüstü) |
| **Tarih** | 03 Mart 2026 |
| **Durum** | Aktif Geliştirme |

---

## 2. Proje Özeti

AI-Sign, **Türk İşaret Dili (TİD)** ile konuşma dili arasında **çift yönlü çeviri** yapabilen yapay zeka tabanlı bir masaüstü uygulamasıdır. Uygulama, kamera aracılığıyla gerçek zamanlı işaret dili tanıma ve metin girişi ile işaret dili video oynatma özelliklerini bir arada sunar.

### Amaç
İşitme engelli bireyler ile işaret dili bilmeyen kişiler arasındaki iletişim engelini yapay zeka teknolojileri kullanarak ortadan kaldırmak.

### Hedef Kitle
- İşitme engelli bireyler
- İşaret dili bilmeyen ancak iletişim kurmak isteyen kişiler
- Eğitim kurumları ve sağlık kuruluşları

---

## 3. Temel Özellikler

### 3.1 İşaret → Metin (Sign-to-Text)
- Kamera üzerinden **gerçek zamanlı** işaret dili tanıma
- MediaPipe Holistic ile vücut, el ve poz analizi
- 3 katmanlı LSTM sinir ağı ile tahmin
- Stabilite filtresi: Son 5 tahminin 4'ü aynı olmalı + **%85 güven eşiği**
- Kullanıcı onay/red mekanizması ile konuşma geçmişi oluşturma

### 3.2 Metin → İşaret (Text-to-Sign)
- Metin girişi ile eşleşen işaret dili videolarının oynatılması
- `raw_videos/` klasöründen kelime bazlı video eşleme
- QTimer tabanlı asenkron video oynatma (UI bloklamaz)

---

## 4. Teknik Mimari

### 4.1 Teknoloji Yığını

| Teknoloji | Kullanım Alanı | Versiyon |
|---|---|---|
| **Python** | Ana programlama dili | 3.x |
| **TensorFlow/Keras** | Derin öğrenme modeli (LSTM) | — |
| **MediaPipe** | İskelet/el/poz algılama | — |
| **OpenCV** | Kamera ve video işleme | — |
| **PyQt5** | Masaüstü arayüz (GUI) | — |
| **NumPy** | Sayısal hesaplama | — |
| **scikit-learn** | Veri bölme (train/val split) | — |

### 4.2 Model Mimarisi

```
Girdi: (30 kare × 258 özellik)
    ↓
LSTM(64, return_sequences=True, activation='relu')
    ↓
LSTM(128, return_sequences=True, activation='relu')
    ↓
LSTM(64, return_sequences=False, activation='relu')
    ↓
Dense(64, activation='relu')
    ↓
Dense(32, activation='relu')
    ↓
Dense(28, activation='softmax')  ← 28 kelime sınıfı
```

- **Girdi vektörü**: 258 boyut (Poz 33×4 + Sol el 21×3 + Sağ el 21×3)
- **Yüz verileri hariç tutulmuştur** (doğruluk artışı için)
- **Normalizasyon**: Burun merkezli + omuz genişliği ile ölçekleme

### 4.3 Veri İşleme Boru Hattı

```
raw_videos/KELIME/*.mp4
        ↓
   process_videos.py  → Uniform Sampling (30 kare)
        ↓                → MediaPipe Keypoint Extraction
   data/KELIME/SEQ/FRAME.npy  (258-dim vektörler)
        ↓
   labels.json güncellenir
        ↓
   train_model.py  → %85/%15 Train/Val Split
        ↓            → EarlyStopping (patience=60)
   action.h5        → En iyi model kaydedilir
```

### 4.4 Dosya Yapısı

```
proje antigravity vson/
├── main.py                  # Ana uygulama (PyQt5 GUI)
├── process_videos.py        # Video → Keypoint dönüşümü
├── train_model.py           # Model eğitimi
├── action.h5                # Eğitilmiş model ağırlıkları
├── labels.json              # Etiket-indeks eşlemesi (28 kelime)
├── requirements.txt         # Python bağımlılıkları
├── AGENTS.md                # Geliştirici kılavuzu
│
├── src/
│   ├── model.py             # LSTM model mimarisi
│   ├── inference.py         # Gerçek zamanlı tahmin motoru
│   ├── keypoints_utils.py   # Keypoint çıkarma (Tek Kaynak)
│   └── avatar_module.py     # Metin→İşaret video modülü
│
├── raw_videos/              # Ham işaret dili videoları (28 kelime)
│   ├── Merhaba/
│   ├── Evet/
│   ├── Hayir/
│   └── ... (28 klasör)
│
├── data/                    # İşlenmiş keypoint verileri
│   └── KELIME/SEQ/FRAME.npy
│
└── _legacy/                 # Eski/kullanılmayan kodlar
```

---

## 5. Desteklenen Kelime Sözlüğü (28 Kelime)

| # | Kelime | # | Kelime | # | Kelime | # | Kelime |
|---|---|---|---|---|---|---|---|
| 0 | Ad | 7 | Evet | 14 | Lutfen | 21 | Sevmek |
| 1 | Araba | 8 | Gel | 15 | Merhaba | 22 | Soyad |
| 2 | Bekle | 9 | Git | 16 | Ne | 23 | Su |
| 3 | Bugun | 10 | Hayir | 17 | Nerede | 24 | Tamam |
| 4 | Doktor | 11 | Iyi | 18 | Okul | 25 | Yardim |
| 5 | Ekmek | 12 | Kac | 19 | Polis | 26 | Yarin |
| 6 | Ev | 13 | Kotu | 20 | Sen | 27 | Zaman |

---

## 6. Uygulama Arayüzü

Uygulama iki ana panelden oluşmaktadır:

### Sol Panel — İşaret → Metin
- **Canlı kamera görüntüsü** (640×480)
- MediaPipe iskelet çizimi üzerinde
- **Öneri kutusu**: Algılanan kelimeyi gösterir
- **Onayla / Reddet** butonları
- **Konuşma geçmişi**: Onaylanan kelimelerin listesi

### Sağ Panel — Metin → İşaret
- **Metin giriş alanı**: Çevrilecek kelime yazılır
- **Çevir ve Oynat** butonu
- **Avatar ekranı** (400×400): İşaret dili videosu oynatılır

---

## 7. Çalıştırma Talimatları

### Gereksinimler
```bash
pip install -r requirements.txt
```

### Yeni Kelime Ekleme
```bash
# 1. Videoları yerleştir
raw_videos/YeniKelime/video1.mp4

# 2. Keypoint çıkar (labels.json otomatik güncellenir)
python process_videos.py

# 3. Modeli yeniden eğit
python train_model.py
```

### Uygulamayı Başlat
```bash
python main.py
```

---

## 8. Teknik Parametreler

| Parametre | Değer | Açıklama |
|---|---|---|
| Kare dizisi uzunluğu | 30 | Her tahmin için 30 karelik pencere |
| Keypoint boyutu | 258 | Poz(132) + Sol el(63) + Sağ el(63) |
| Güven eşiği | %85 | Tahminin kabul edilme minimum olasılığı |
| Stabilite kriteri | 5 tahminden 4'ü | Art arda tutarlılık kontrolü |
| Eğitim/Doğrulama bölümü | %85 / %15 | Stratified split |
| Maksimum epoch | 300 | EarlyStopping ile erken durdurma |
| EarlyStopping patience | 60 | 60 epoch iyileşme olmazsa dur |
| Optimizer | Adam | Adaptif öğrenme hızı |
| Kayıp fonksiyonu | Categorical Crossentropy | Çok sınıflı sınıflandırma |

---

## 9. Riskler ve Kısıtlamalar

| Risk / Kısıtlama | Açıklama |
|---|---|
| Sınırlı kelime sözlüğü | Şu anda 28 kelime desteklenmektedir |
| Tek kelime tanıma | Cümle düzeyinde çeviri henüz desteklenmemektedir |
| Aydınlatma bağımlılığı | Kamera performansı ortam ışığına bağlıdır |
| Kişi bağımlılığı | Model, eğitim verisindeki kişilere daha hassastır |
| Tek kamera | Yalnızca ön kamera ile çalışır (derinlik sensörü yok) |

---

## 10. Gelecek Geliştirme Planları

- [ ] Kelime sözlüğünün genişletilmesi (50+ kelime)
- [ ] Cümle düzeyinde çeviri desteği
- [ ] Çoklu kullanıcı eğitimi ile genelleme iyileştirmesi
- [ ] Mobil platform desteği (Android/iOS)
- [ ] Sesli çıktı entegrasyonu (TTS)
- [ ] Bulut tabanlı model sunumu (API)
- [ ] Parmak hareketleri için detaylı el izleme

---

> **Not**: Bu belge, projenin mevcut durumunu yansıtmaktadır (03 Mart 2026). Güncellemeler için proje deposundaki kaynak kodları incelenmelidir.
