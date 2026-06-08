---
name: Code Quality Guidelines
description: Kod kalitesi, loglama, hata yönetimi ve docstring kuralları
---

# Code Quality Guidelines

## 1. Typed Error Handling ve Log Yönetimi
- Hata yönetimi jenerik bir `except Exception:` ile geçiştirilemez. Hataya özel tipler (Typed Exception handling) yakalanmalıdır (Örn: `except ValueError:`).
- `print()` yerine merkezi bir loglama kütüphanesi (örn: Python'da `logging` veya `loguru`) kullanılmalıdır.
- Loglar sadece hata ayıklamak için değil, uygulamanın kritik durumları hakkında yapılandırılmış (structured logging, JSON log) bilgi vermek için kullanılmalıdır (`info`, `warning`, `error`, `critical`).
- İlgili hatalar (`TRY` kuralları ruff ile zorunludur) loglanıp uygun seviyede yönetilmelidir.

## 2. Ortak İşlevlerin Modüler Hale Getirilmesi (Utils/Helpers)
- Tekrar eden, farklı servisler tarafından çağrılan işlevler genelleştirilip `utils/` veya `helpers/` dizinine alınmalıdır.
- Utils fonksiyonları saf (pure) olmalı; dışarıdaki state'i (global state) değiştirmemeli ve her çağrıldığında öngörülebilir sonuçlar üretmelidir.

## 3. Docstring ve Yorum (Comment) Kuralları
- Tüm sınıf, metot ve önemli modüller açık ve anlaşılır şekilde Google veya NumPy style docstring formatında dokümante edilmelidir (Ruff `D` kuralları aktiftir).
- Ne yapıldığı (What) veya kodun kendisi değil, **neden (Why)** yapıldığı yorum satırlarında (inline comment) belirtilmelidir.
- Type Hint (Type Annotations) kullanımı kod okunabilirliğini sağlamak amacıyla Python'da zorunludur (örneğin `def process(data: str) -> bool:`).
