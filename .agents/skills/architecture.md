---
name: Architecture Guidelines
description: Temel mimari kurallar (SOLID, DRY, Async-first, DI, Config)
---

# Architecture Guidelines for Antigravity IDE (Python)

Bu dosya, projede bulunan tüm geliştiriciler/ajanlar için uyulması zorunlu temel mimari kuralları belirler.

## 1. SOLID ve DRY Prensipleri
- **Single Responsibility (SRP):** Her sınıfın ve modülün tek bir sorumluluğu olmalıdır.
- **Open/Closed (OCP):** Sınıflar gelişime açık, değişime kapalı olmalıdır.
- **DRY (Don't Repeat Yourself):** Aynı kod bloğunu tekrarlamak yerine modüler, yeniden kullanılabilir yapılar (fonksiyon/sınıf) oluşturun.

## 2. GUI ve Ağır İşlemlerin İzolasyonu (Kritik UI Kuralı)
- **PyQt5 ve MediaPipe Çakışması:** Sistemde işaret dili tercümanı gibi eşzamanlı, ağır görüntü işleme (landmark extraction) süreçleri mevcuttur. Python `asyncio` yapısı PyQt5 tabanlı bir GUI'de bu CPU bağımlı yükü asenkron yapmaya yetmez, uygulama donar/kilitlenir.
- Arayüzü asmak kesinlikle yasaktır! Görüntü işleme ve yapay zeka çıkarımı (AI Inference) yapan döngüler izolasyon amacıyla **kesinlikle `QThread`** içine alınmalıdır.
- Arka plan iş parçacıklarından (worker) arayüze veri aktarmak için (örneğin kameradan alınan frame'i ekranda göstermek) **yalnızca `pyqtSignal`** kullanılmalıdır. Diğer yöntemler thread-safe değildir.
- I/O işlemleri (HTTP istekleri vb.) için standart `async/await` kullanılabilir, ancak AI loop'ları `QThread` kuralına bütünüyle tabidir.

## 3. Dependency Injection (DI) Kullanımı
- Klasik Singleton (Global state) yerine Dependency Injection (Bağımlılık Enjeksiyonu) kullanılmalıdır.
- Sınıfların bağımlılıkları `__init__` üzerinden verilmeli, sınıfın kendisi nesneyi (instance) yaratmamalıdır.
- (Scoped/Transient/Singleton) yaşam döngüsü senaryolarına uygun container kütüphaneleri (örneğin Python'da `dependency_injector` veya `punq`) tercih edilmeli.

## 4. Konfigürasyon Yönetimi (Hardcode Yasak)
- API anahtarları, DB bağlantı adresleri, portlar gibi çevresel değişkenler (environment variables) veya dış vault sistemlerinden (HashiCorp Vault vb.) okunmalıdır.
- Kod içinde hardcode sabitler bulundurmak kesinlikle yasaktır.
- Python konfigürasyonu için `.env` (örneğin `python-dotenv`) ve `pydantic-settings` kullanılmalıdır.
