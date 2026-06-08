# Katkıda Bulunma Rehberi (Contributing)

AI-Sign projesine ilgi gösterdiğiniz için teşekkür ederiz! Bu projeyi açık kaynak olarak geliştirmekten mutluluk duyuyoruz. Projeye kod, dokümantasyon, hata bildirimi veya yeni işaret dili kelimeleri ekleyerek katkıda bulunabilirsiniz.

## Nasıl Katkıda Bulunabilirsiniz?

### 1. Hata Bildirimi (Bug Report)
Bir sorunla karşılaşırsanız, lütfen GitHub "Issues" bölümünden **Bug Report** şablonunu kullanarak bizimle paylaşın. Hatayı tekrar edebilmemiz için adım adım ne yaptığınızı yazmanız çok önemlidir.

### 2. Yeni Özellik İsteği (Feature Request)
Projeye eklenebilecek yeni bir fikriniz varsa, "Issues" altından **Feature Request** şablonunu kullanarak bize detaylı bir şekilde açıklayın.

### 3. Kod Katkısı (Pull Request)
Eğer koda doğrudan katkı sağlamak istiyorsanız:
1. Projeyi kendi GitHub hesabınıza **fork** edin.
2. Yeni bir dal (branch) açın: `git checkout -b feature/ozellik-adiniz` veya `git checkout -b fix/hata-adiniz`.
3. Kodunuzu yazın. (Lütfen mevcut kodlama standartlarına ve mimariye uymaya özen gösterin).
4. Değişikliklerinizi kaydedin (commit): `git commit -m "feat: Yeni özellik eklendi"`.
5. Dalınızı GitHub'a gönderin (push): `git push origin feature/ozellik-adiniz`.
6. Orijinal repoya bir **Pull Request (PR)** açın.

### 4. Yeni Kelime / Veri Ekleme
Projeye yeni işaret dili kelimeleri eklemek isterseniz:
1. `raw_videos/` altına yeni kelime adıyla bir klasör açıp videolarınızı koyun.
2. `process_videos.py` betiğini çalıştırarak veri noktalarını (keypoints) çıkarın (Bu işlem `labels.json`'ı da otomatik günceller).
3. Modeli yeniden eğitmek için `train_model.py` dosyasını çalıştırın.
4. Çıkan yeni `action.tflite` modelini ve güncel `labels.json`'ı Pull Request ile gönderebilirsiniz.

Katkılarınız için şimdiden teşekkürler! 🚀
