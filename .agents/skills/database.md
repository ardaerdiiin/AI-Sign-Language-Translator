---
name: Database Guidelines
description: Veritabanı tasarım ve sorgulama standartları
---

# Database Guidelines

## 1. Minimalizm ve İhtiyaca Göre Ölçekleme (Anti Over-Engineering)
- Uygulama otonom/yerel ağırlıklı çalışan bir yapay zeka işaret dili tercümanıdır. Dev bulut veritabanlarına veya karmaşık RDBMS mimarilerine (kümelenmiş sunuculara) ihtiyaç kalmadıkça, mimariyi şişirmek "over-engineering" tuzağıdır.
- Yerel çeviri logları, kullanıcı tercihleri ve geçmiş profilleri için **SQLite** gibi hafif ve dosyaya gömülü (embedded) veri tabanları kullanılmalıdır.

## 2. Primary Key Stratejisi
- Yüksek eşzamanlı yazma gerektiren (write-heavy) bir cloud altyapısı kurulana kadar `Sequential UUID` veya benzeri kompleks indeksleme zorunlu değildir.
- SQLite kullanımında basit okuma/yazma kolaylığı adına Standart Auto-Increment veya `UUIDv4` yeterlidir.

## 3. ORM Kullanımı ve Veri Çekme
- ORM entegrasyonunda N+1 query dikkat edilmesi gereken bir performans detayıdır; fakat karmaşık Eager-Loading (örn: `joinedload` vb.) planlaması yerine, küçük yerel veritabanı kullanımında **okunabilirlik ve sadelik** daha ön plandadır.
- Sadece ihtiyaç duyulan alanların istenmesi yeterlidir.

## 4. Kayıt Silme İşlemleri
- Çok hayati ve muhasebe gereksinimi olan kullanıcı verileri hariç olmak üzere, basit ara loglar veya geçici cache tablolarında 'Soft Delete' (`IsDeleted`) bayrağı taşıttırmak veritabanını boş yere şişirebilir. İhtiyaca göre doğrudan `DELETE` işlemi (Hard Delete) yapılabilir.
