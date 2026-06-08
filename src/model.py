"""
Transformer Encoder modeli — TSLFormer mimarisinden ilham alınmıştır.

Makale: "TSLFormer: A Lightweight Transformer Model for Turkish Sign Language
         Recognition Using Skeletal Landmarks" (arXiv:2505.07890v4)

Mimari:
  Input (SEQUENCE_LENGTH, 194)
    → Dense Embedding (194 → 512)
    → Sinüzoidal Positional Encoding
    → Transformer Encoder × 2 katman (4-head self-attention)
    → Mean Pooling
    → Dense(128) → Dropout → Dense(softmax)

Parametreler (makaledeki ile aynı):
  d_model  = 512
  nhead    = 4
  num_layers = 2
  dropout  = 0.2
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Dropout, LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D, Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ============================================================================
# Positional Encoding (Sinüzoidal — "Attention Is All You Need")
# ============================================================================

class PositionalEncoding(tf.keras.layers.Layer):
    """
    Transformer ağlarında verilerin zaman sırasını (frame sırasını) anlamlandırmak için kullanılan 
    pozisyonel kodlama katmanı. LSTM'lerin aksine Transformer'lar tüm diziyi aynı anda işler. 
    Bu yüzden hangi karenin önce, hangisinin sonra geldiğini modele öğretmek için bu katmanda
    hareket dizisine matematiksel sinüs ve kosinüs dalgaları eklenir.
    """

    def __init__(self, max_len=200, d_model=512, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model

        # Pozisyonel kodlama matrisini önceden hesapla (eğitilebilir değil)
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        # (1, max_len, d_model) — batch boyutu için broadcast
        self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_len": self.max_len,
            "d_model": self.d_model,
        })
        return config


# ============================================================================
# Transformer Encoder Bloğu
# ============================================================================

def _transformer_encoder_block(x, d_model, nhead, ff_dim, dropout_rate):
    """
    Modelin temel yapı taşı olan Tekli Transformer Encoder bloğu.
    Kendi içindeki farklı karelerin birbirleriyle olan ilişkisini analiz eder.
    İçeriği: Multi-Head Self-Attention (Çoklu Dikkat) → Add & Norm → FeedForward → Add & Norm
    """
    # Multi-Head Self-Attention
    attn_output = MultiHeadAttention(
        num_heads=nhead,
        key_dim=d_model // nhead,
        dropout=dropout_rate,
    )(x, x)
    attn_output = Dropout(dropout_rate)(attn_output)
    x1 = LayerNormalization(epsilon=1e-6)(x + attn_output)

    # Position-wise Feed-Forward Network
    ff_output = Dense(ff_dim, activation='relu')(x1)
    ff_output = Dropout(dropout_rate)(ff_output)
    ff_output = Dense(d_model)(ff_output)
    ff_output = Dropout(dropout_rate)(ff_output)
    x2 = LayerNormalization(epsilon=1e-6)(x1 + ff_output)

    return x2


# ============================================================================
# Model Oluşturucu
# ============================================================================

# Makaledeki sabitler
_D_MODEL = 512
_NHEAD = 4
_NUM_LAYERS = 2
_FF_DIM = 1024       # Feed-forward ara boyutu (genellikle 2×d_model veya 4×d_model)
_DROPOUT = 0.2
_SEQUENCE_LENGTH = 16
_FEATURE_DIM = 194   # Mühendislik özellik vektörü boyutu

def build_model(output_units):
    """
    Keras API kullanılarak uçtan uca Transformer modelini inşa eder ve derler.
    Model Eğitim dosyasında (train_model.py) ağırlıkları rastgele başlatmak ve eğitmek için kullanılır.

    Model Akışı:
    1. Input Katmanı (16 kare x 194 özellik)
    2. Özellikleri 512 boyuta genişletme (Dense Embedding)
    3. Positional Encoding (Zaman sırasını ekleme)
    4. 2 adet Transformer Encoder Bloğu
    5. Tüm zaman adımlarının ortalamasını alma (GlobalAveragePooling1D)
    6. Tam Bağlı (Dense) katmanlar ile kelime tahmini (Softmax Çıkışı)

    Args:
        output_units: Modelin sınıflandıracağı toplam kelime sayısı (labels.json uzunluğu)
    Returns:
        Derlenmiş (Adam optimizer ve Categorical Crossentropy) tf.keras.Model objesi.
    """
    # --- Giriş ---
    inputs = Input(shape=(_SEQUENCE_LENGTH, _FEATURE_DIM))

    # --- Embedding: 194 → 512 boyutlu temsil ---
    x = Dense(_D_MODEL, activation='relu')(inputs)

    # --- Positional Encoding ---
    x = PositionalEncoding(max_len=_SEQUENCE_LENGTH, d_model=_D_MODEL)(x)
    x = Dropout(_DROPOUT)(x)

    # --- Transformer Encoder Katmanları ---
    for _ in range(_NUM_LAYERS):
        x = _transformer_encoder_block(x, _D_MODEL, _NHEAD, _FF_DIM, _DROPOUT)

    # --- Mean Pooling: Tüm karelerin ortalaması → tek vektör ---
    x = GlobalAveragePooling1D()(x)

    # --- Sınıflandırma Kafası ---
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(output_units, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)

    # --- Optimizer (makaledeki ile aynı: Adam, lr=1e-4) ---
    optimizer = Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy'],
    )
    return model
