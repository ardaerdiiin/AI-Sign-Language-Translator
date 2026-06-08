---
description: Yeni bir özellik eklerken izlenecek adım adım mikro görev listesi.
---

# Feature Workflow

Yeni bir özellik (feature) geliştirirken odaklanmayı kaybetmemek ve projeyi yarım bırakmamak için aşağıdaki adımları sırayla izleyin:

1. **İhtiyaç Analizi ve Tasarım:**
   - Eklenecek özelliğin neleri kapsadığını (ve neleri kapsamadığını) netleştirin.
   - İhtiyaç duyulan değişikliklerin mimari kurallarına (`architecture.md`, `database.md`) uyduğundan emin olun.

2. **Branch Stratejisi:**
   - Özellik için spesifik, kısa ömürlü bir feature branch oluşturun.
   - Örn: `feature/avatar-yeni-animasyon`

3. **Veri / DB Değişiklikleri:**
   - Veritabanı modeli eklenecekse, migration dosyasını oluşturun (Sequential UUID ve Soft Delete dahil).
   - Veritabanı değişikliklerini test ortamında uygulayın.

4. **Çekirdek (Core / Domain) İş Mantığı:**
   - Asenkron (async-first) metotlarla gerekli servisi veya util fonksiyonunu yazın.
   - İlgili logic için Modüler fonksiyonlar (utils) oluşturun.

5. **Test Yazımı (TDD veya Sonradan Ekleme):**
   - Mutlaka minimum bir Happy-Path ve Error-Handling senaryosu içeren test tasarlayın. 

6. **Entegrasyon ve Arayüz (UI / API):**
   - Özelliği uygulamanın geneliyle (UI veya API endpoint) bağlayın.
   - Eğer yeni API/UI eklendiyse, N+1 query üretip üretmediğini veya UI'yi bloklayıp (async ihlali) asmadığını kontrol edin.

7. **Code Review ve Refactor:**
   - Kodu check-in yapmadan önce `code-review.md` checklist'ini tamamlayın.
