import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path

# DB file will be stored at project root
DB_PATH = Path(__file__).resolve().parent.parent / "database.sqlite"

def init_db():
    """
    Uygulamanın SQLite veritabanını (database.sqlite) ve tablolarını (users, feedbacks) oluşturur.
    'users': Kullanıcı hesaplarını (kullanıcı adı ve şifre) güvenli tutar.
    'feedbacks': Kullanıcıların tahminlere verdiği Doğru/Yanlış (Aktif Öğrenme) tepkilerini kaydeder.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create feedbacks table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedbacks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            predicted_word TEXT NOT NULL,
            is_correct BOOLEAN NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    
    conn.commit()
    conn.close()

def _hash_password(password):
    """
    Güvenlik amacıyla kullanıcı şifrelerini veritabanına düz metin (plain text) olarak 
    kaydetmek yerine SHA-256 algoritmasıyla şifreleyerek (hash) döndürür.
    """
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    """
    Yeni kullanıcı kaydını gerçekleştirir. Kullanıcı adı ve şifrenin boş olup olmadığını kontrol eder.
    Şifreyi hash'leyerek veritabanına yazar. Aynı kullanıcı adı varsa hata döndürür.
    """
    if not username or not password:
        return False, "Kullanıcı adı veya şifre boş olamaz."
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", 
                       (username, _hash_password(password)))
        conn.commit()
        return True, "Kayıt başarılı."
    except sqlite3.IntegrityError:
        return False, "Bu kullanıcı adı zaten mevcut."
    finally:
        conn.close()

def authenticate_user(username, password):
    """
    Kullanıcı girişini (Login) doğrular. Girilen şifrenin hash'i ile veritabanındaki hash'i karşılaştırır.
    Giriş başarılıysa kullanıcının ID'sini döndürür (Geri bildirimleri kişiselleştirmek için).
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id FROM users WHERE username = ? AND password_hash = ?", 
                   (username, _hash_password(password)))
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return True, user[0] # Return user_id
    else:
        return False, None

def log_feedback(user_id, predicted_word, is_correct):
    """
    Kullanıcının arayüzde 'Onayla' (Doğru) veya 'Reddet' (Yanlış) butonuna 
    bastığı durumları veritabanındaki 'feedbacks' tablosuna kaydeder.
    Bu veriler, modeli gelecekte yeniden eğitmek veya ince ayar (fine-tune) yapmak için kullanılabilir.
    """
    if not user_id:
        return
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO feedbacks (user_id, predicted_word, is_correct) VALUES (?, ?, ?)",
                   (user_id, predicted_word, is_correct))
    conn.commit()
    conn.close()
