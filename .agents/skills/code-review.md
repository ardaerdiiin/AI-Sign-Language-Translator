---
name: Code Review Checklist
description: Teknik borç yaratmamak için check-in öncesi kontrol listesi.
---

# Code Review Checklist (Pre Check-in)

Pull Request açılmadan veya Check-in yapılmadan önce geliştiriciler aşağıdaki öğeleri doğrulamalıdır:

## 1. İsimlendirme ve Standartlar (Naming Convention)
- Değişken ve fonksiyon isimleri açıklayıcı mı? (Python için `snake_case`, Sınıflar için `PascalCase` prensibine uyuldu mu?)
- Boolean değişkenler isimlendirilirken soru belirtiyor mu? (`is_active`, `has_permission` v.b.)

## 2. Mimari Kurallar
- Modül veya Fonksiyon Single Responsibility'ye uyuyor mu? (SOLID)
- Yeterince küçük metotlar haline ayrıldı mı? (Complexity <= 8)
- I/O işlemleri `async/await` mimarisine uygun mu? Hardcode string veya config bırakılmış mı? (Vault/Env zorunluluğu)

## 3. Hata Yönetimi
- Catch-all exception ( `except Exception:` ) yerine spesifik hata sınıfları hedeflendi mi?
- Açıklayıcı hata mesajları ve doğru seviye (WARN, ERROR vs) loglama (Structured log) eklendi mi?

## 4. Veritabanı Performansı
- Yeni eklenen sorgularda N+1 problemi tetikleniyor mu? (Örn: döngü içinde DB sorgusu)
- Kullanılan kalıcı silme varsa "Soft Delete" yapısına (`IsDeleted` / `DeletedAt`) dönüştürüldü mü?

## 5. Dokümantasyon
- Beklenmedik "Hack" veya çok teknik bir workaround yazıldıysa **nedeni** comment ile açıklandı mı?
- Sınıf ve modül seviyesi docstring'ler güncellendi mi?

> **Tüm maddeler "Evet" veya "Uygulanamaz" ise kod check-in için hazırdır.**
