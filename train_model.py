import numpy as np
import os
import json
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split 
from keras.utils import to_categorical
from src.model import build_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# --- Yapılandırma (Mutlak Yollar) ---
_PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = str(_PROJECT_ROOT / 'data')
LABELS_PATH = str(_PROJECT_ROOT / 'labels.json')
MODEL_PATH = str(_PROJECT_ROOT / 'action.h5')
SEQUENCE_LENGTH = 16  # Transformer Modelin beklediği kare sayısı (TSLFormer ile aynı)


def load_labels():
    """labels.json'dan deterministik etiket sırasını okur."""
    if not os.path.exists(LABELS_PATH):
        print(f"Hata: {LABELS_PATH} bulunamadı! Önce process_videos.py çalıştırın.")
        sys.exit(1)
    with open(LABELS_PATH, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    # İndekse göre sırala → dizi
    sorted_labels = sorted(label_map.items(), key=lambda x: x[1])
    return np.array([name for name, _ in sorted_labels]), label_map


def augment_keypoint_data(X, y, multiplier=5, noise_factor=0.02):
    """
    Eğitim verisini sentetik olarak çoğaltır (Data Augmentation).
    Mevcut iskelet koordinatlarına (X) rastgele ufak gürültüler (noise) ekleyerek
    hareketin farklı açılardan veya hafif titreyerek yapılmış alternatif versiyonlarını üretir.
    Bu, modelin ezberlemesini (overfitting) engeller ve genelleme yeteneğini artırır.
    """
    print(f"Veri çoğaltılıyor... Orijinal veri boyutu: {len(X)}")
    X = X.astype(np.float32)
    X_aug, y_aug = [X], [y]
    
    for _ in range(multiplier - 1): # Orijinal veri 1. kopya sayılır, (multiplier-1) kadar yeni üretilir
        # Keypoint koordinatlarına rastgele küçük gürültüler ekle
        noise = np.random.normal(0, noise_factor, X.shape)
        X_noisy = X + noise
        
        X_aug.append(X_noisy)
        y_aug.append(y)
        
    X_combined = np.concatenate(X_aug)
    y_combined = np.concatenate(y_aug)
    print(f"Çoğaltılmış yeni veri boyutu: {len(X_combined)}")
    return X_combined, y_combined


def load_data(actions, label_map):
    """
    Önceden işlenip 'data' klasörüne kaydedilen (process_videos.py) '.npy' verilerini okur.
    Her bir video dizisi (16 kare), modele girdi (X) olarak eklenir. 
    Karşılık gelen kelime (Y etiketi) ise makinenin anlayabileceği "One-Hot Encoding"
    formatına (to_categorical) çevrilir.
    """
    sequences, labels = [], []
    for action in actions:
        action_idx = label_map[action]
        print(f"Loading data for: {action} (idx={action_idx})")
        try:
            action_dir = os.path.join(DATA_PATH, action)
            if not os.path.isdir(action_dir):
                print(f"  Uyarı: {action_dir} klasörü yok, atlanıyor.")
                continue

            sequences_dirs = [d for d in os.listdir(action_dir)
                              if os.path.isdir(os.path.join(action_dir, d))]

            for sequence in sequences_dirs:
                window = []
                is_valid_sequence = True
                for frame_num in range(SEQUENCE_LENGTH):
                    res_path = os.path.join(DATA_PATH, action, sequence, f"{frame_num}.npy")
                    if not os.path.exists(res_path):
                        is_valid_sequence = False
                        break
                    res = np.load(res_path)
                    window.append(res)

                if is_valid_sequence:
                    sequences.append(window)
                    labels.append(action_idx)
        except Exception as e:
            print(f"Error loading {action}: {e}")
            sys.exit(1)

    X = np.array(sequences)
    y = to_categorical(labels, num_classes=len(actions)).astype(int)
    return X, y, labels


def main():
    """
    Transformer AI modelinin ana eğitim döngüsüdür (Training Pipeline).
    Sırasıyla şunları yapar:
    1. Etiketleri ve NumPy dizilerini (LSTM verisi) belleğe alır.
    2. Verisetini %85 Eğitim (Train) ve %15 Test (Validation) olarak ayırır.
    3. Sadece eğitim verisini çoğaltarak veri sayısını artırır (Augmentation).
    4. Ortamda önceden eğitilmiş 'action.h5' dosyası varsa onu yükleyip kaldığı yerden devam eder (Incremental Fine-tuning), yoksa sıfırdan model oluşturur.
    5. Modeli eğitir. Başarı arttıkça en iyi ağırlıkları kaydeder (ModelCheckpoint) ve 25 adım (epoch) boyunca gelişme olmazsa eğitimi keser (EarlyStopping).
    6. Eğitim sonunda başarı oranlarını (Classification Report) ve Confusion Matrix ısı haritasını png olarak kaydeder.
    """
    print("Etiketler yükleniyor...")
    actions, label_map = load_labels()
    print(f"Toplam {len(actions)} etiket: {list(actions)}")

    print("Veriler yükleniyor...")
    X, y, raw_labels = load_data(actions, label_map)

    # --- Train/Validation Split ---
    print(f"Toplam örnek: {len(X)}")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.15,
        random_state=42
    )
    print(f"Train (Bölünme sonrası): {len(X_train)}, Validation: {len(X_val)}")
    
    # AUTSL zaten 2700+ video, hafif augmentation yeterli
    X_train, y_train = augment_keypoint_data(X_train, y_train, multiplier=3, noise_factor=0.02)
    print(f"Train (Artırım sonrası): {len(X_train)}")

    # Eski ağırlıklar kontrolü ve transfer learning (Incremental Fine-tuning)
    if os.path.exists(MODEL_PATH):
        try:
            from src.model import PositionalEncoding
            from tensorflow.keras.models import load_model
            model = load_model(MODEL_PATH, custom_objects={'PositionalEncoding': PositionalEncoding})
            print(f"Eski model yuklendi: {MODEL_PATH}. Mevcut agirliklarin uzerine egitim yapilacak (Incremental Fine-tuning).")
            
            # Etiket sayısı değiştiyse (yeni kelime eklendiyse) modelin son katmanını güncelle
            if model.output_shape[-1] != len(actions):
                print(f"Uyarı: Eski modeldeki etiket sayısı ({model.output_shape[-1]}) yeni etiket sayısıyla ({len(actions)}) uyuşmuyor!")
                print("Modelin son katmanı yeni kelimeler için yeniden yapılandırılıyor (Transfer Learning)...")
                from tensorflow.keras.layers import Dense
                from tensorflow.keras.models import Model
                from tensorflow.keras.optimizers import Adam
                # Son katmanı (Dense) at, yerine yeni boyutlu Dense ekle
                new_output = Dense(len(actions), activation='softmax')(model.layers[-2].output)
                model = Model(inputs=model.input, outputs=new_output)
                model.compile(
                    optimizer=Adam(learning_rate=1e-4),
                    loss='categorical_crossentropy',
                    metrics=['categorical_accuracy'],
                )
        except Exception as e:
            print(f"Eski model yuklenemedi veya uyumsuz ({e}). Sifirdan model olusturuluyor...")
            model = build_model(len(actions))
    else:
        model = build_model(len(actions))

    print("Model eğitimi başlıyor...")

    # Val accuracy monitörleme — gerçek performans metriği
    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor='val_categorical_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max',
    )

    early_stopping = EarlyStopping(
        monitor='val_categorical_accuracy',
        patience=25,
        restore_best_weights=True,
        verbose=1,
    )

    model.fit(
        X_train, y_train,
        epochs=300,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping],
    )

    # --- Değerlendirme ve Confusion Matrix ---
    print("Değerlendirme yapılıyor...")
    from sklearn.metrics import confusion_matrix, classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns

    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)

    print("\nSınıflandırma Raporu:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=actions))

    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=actions, yticklabels=actions)
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("confusion_matrix.png kaydedildi.")

    print("Eğitim Tamamlandı.")

    # Yeni Videoları Asıl Klasöre Taşı (Merge)
    import shutil
    import uuid
    NEW_RAW_VIDEOS_PATH = str(_PROJECT_ROOT / 'new_raw_videos')
    RAW_VIDEOS_PATH = str(_PROJECT_ROOT / 'raw_videos')
    
    if os.path.exists(NEW_RAW_VIDEOS_PATH) and len(os.listdir(NEW_RAW_VIDEOS_PATH)) > 0:
        print("\nYeni videolar ana veri setine (raw_videos) taşınıyor...")
        for action_name in os.listdir(NEW_RAW_VIDEOS_PATH):
            src_action_dir = os.path.join(NEW_RAW_VIDEOS_PATH, action_name)
            if not os.path.isdir(src_action_dir):
                continue
                
            dest_action_dir = os.path.join(RAW_VIDEOS_PATH, action_name)
            if not os.path.exists(dest_action_dir):
                os.makedirs(dest_action_dir)
                
            for video_file in os.listdir(src_action_dir):
                src_video = os.path.join(src_action_dir, video_file)
                dest_video = os.path.join(dest_action_dir, video_file)
                
                # Eğer aynı isimde video varsa çakışmayı önlemek için ismini değiştir
                if os.path.exists(dest_video):
                    base, ext = os.path.splitext(video_file)
                    unique_id = uuid.uuid4().hex[:6]
                    dest_video = os.path.join(dest_action_dir, f"{base}_{unique_id}{ext}")
                    
                shutil.move(src_video, dest_video)
                print(f"  Taşındı: {video_file} -> {dest_action_dir}")
                
            # İçi boşalan action klasörünü sil
            shutil.rmtree(src_action_dir)
            
        print("Tüm yeni videolar başarıyla taşındı. 'new_raw_videos' klasörü boşaltıldı.")


if __name__ == '__main__':
    main()
