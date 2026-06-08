"""
Keras H5 Modelini TFLite Formatına Dönüştürücü
================================================
action.h5 → action.tflite

Kullanım:
    python convert_to_tflite.py

Model eğitildikten sonra (train_model.py) bu scripti çalıştırın.
TFLite modeli ~7x daha hızlı inference sağlar (20ms → ~3ms).
"""

import os
import sys
from pathlib import Path

# Proje kök dizini
_PROJECT_ROOT = Path(__file__).resolve().parent
H5_PATH = str(_PROJECT_ROOT / 'action.h5')
TFLITE_PATH = str(_PROJECT_ROOT / 'action.tflite')


def convert():
    """
    Eğitilmiş Keras (.h5) modelini okur ve TensorFlow Lite (.tflite) formatına dönüştürür.
    Bunu yaparken modeli 'Float16 Quantization' yöntemiyle sıkıştırır.
    Bu işlem, modelin hem diskte kapladığı alanı yarı yarıya azaltır hem de 
    canlı kameradan tahmin yaparken (inference) hızını ciddi şekilde artırır (~3ms).
    """
    if not os.path.exists(H5_PATH):
        print(f"Hata: {H5_PATH} bulunamadi!")
        print("Once train_model.py ile modeli egit.")
        sys.exit(1)

    # TensorFlow'u burada import et (yükleme süresi uzun)
    import tensorflow as tf

    print(f"Model yukleniyor: {H5_PATH}")
    from src.model import PositionalEncoding
    model = tf.keras.models.load_model(
        H5_PATH,
        custom_objects={'PositionalEncoding': PositionalEncoding},
    )
    model.summary()

    print("\nTFLite'a donusturuluyor...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Float16 quantization — boyutu yarıya düşürür, hız artırır
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    # Kaydet
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)

    h5_size = os.path.getsize(H5_PATH) / 1024
    tflite_size = len(tflite_model) / 1024
    print(f"\nDonusum tamamlandi!")
    print(f"  H5     : {h5_size:.1f} KB")
    print(f"  TFLite : {tflite_size:.1f} KB ({tflite_size/h5_size*100:.0f}%)")
    print(f"  Kaydedildi: {TFLITE_PATH}")


if __name__ == '__main__':
    convert()
