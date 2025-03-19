import os
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import pickle
from sklearn.neighbors import KNeighborsClassifier

# 📂 Kayıtların saklanacağı klasör ve model dosyası
VERI_KLASORU = "dataset"
MODEL_DOSYA = "voice_model.pkl"

if not os.path.exists(VERI_KLASORU):
    os.makedirs(VERI_KLASORU)

# 🎤 Ses kayıt ayarları
SURE = 3  # Kayıt süresi (saniye)
ORNEKLEME_ORANI = 22050  # Örnekleme frekansı

# 📌 Modeli yükle veya yeni bir model oluştur
def model_yukle_veya_olustur():
    if os.path.exists(MODEL_DOSYA):
        with open(MODEL_DOSYA, "rb") as f:
            model, ogrenci_verileri = pickle.load(f)
        print("✅ Model başarıyla yüklendi!")
    else:
        model = KNeighborsClassifier(n_neighbors=3)  # KNN Modeli
        ogrenci_verileri = {}  # Öğrenci bilgilerini saklayan sözlük
        print("⚙️ Model sıfırdan oluşturuluyor!")
    return model, ogrenci_verileri

model, ogrenci_verileri = model_yukle_veya_olustur()

def ses_kaydet(dosya_adi):
    """ 📢 Ses kaydı yap ve dosyaya kaydet """
    try:
        print(f"🎤 {SURE} saniye boyunca konuşun...")
        ses_verisi = sd.rec(int(SURE * ORNEKLEME_ORANI), samplerate=ORNEKLEME_ORANI, channels=1, dtype='float32')
        sd.wait()
        sf.write(dosya_adi, ses_verisi, ORNEKLEME_ORANI)
        print("✅ Ses kaydı tamamlandı!")
    except Exception as e:
        print(f"❌ Ses kaydı hatası: {e}")

def ozellikleri_cikar(dosya_yolu):
    """ 🔍 Ses dosyasından MFCC özelliklerini çıkar """
    try:
        y, sr = librosa.load(dosya_yolu, sr=ORNEKLEME_ORANI, mono=True)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        return np.mean(mfccs, axis=1)
    except Exception as e:
        print(f"❌ Ses özellikleri çıkarılamadı: {e}")
        return None

def ogrenci_kayit():
    """ 📝 Yeni öğrenci kaydı al """
    isim = input("👤 Öğrencinin adını girin: ").strip()
    ogrenci_no = input("🔢 Öğrenci numarasını girin: ").strip()

    dosya_yolu = os.path.join(VERI_KLASORU, f"{ogrenci_no}.wav")
    ses_kaydet(dosya_yolu)

    ozellikler = ozellikleri_cikar(dosya_yolu)
    if ozellikler is None:
        print("❌ Özellik çıkarma başarısız!")
        return

    ogrenci_verileri[ogrenci_no] = isim

    if hasattr(model, "fit"):
        X, y = [], []
        for numara, ad in ogrenci_verileri.items():
            veri_dosyasi = os.path.join(VERI_KLASORU, f"{numara}.wav")
            ozellikler = ozellikleri_cikar(veri_dosyasi)
            if ozellikler is not None:
                X.append(ozellikler)
                y.append(numara)

        if X and y:
            model.fit(X, y)

            with open(MODEL_DOSYA, "wb") as f:
                pickle.dump((model, ogrenci_verileri), f)

        print(f"✅ {isim} ({ogrenci_no}) başarıyla kaydedildi!")
    else:
        print("❌ Model güncellenemedi!")

def ogrenci_sil():
    """ 🧹 Öğrenci kaydını sil """
    ogrenci_no = input("🔢 Silmek istediğiniz öğrenci numarasını girin: ").strip()

    if ogrenci_no in ogrenci_verileri:
        del ogrenci_verileri[ogrenci_no]
        dosya_yolu = os.path.join(VERI_KLASORU, f"{ogrenci_no}.wav")
        if os.path.exists(dosya_yolu):
            os.remove(dosya_yolu)
            print(f"✅ {ogrenci_no} numaralı öğrenci ve kaydı başarıyla silindi!")
        else:
            print("❌ Öğrenci kaydına ait ses dosyası bulunamadı!")
    else:
        print("❌ Öğrenci numarası bulunamadı!")

def yoklama_al():
    """ 📋 Yoklama al ve öğrenciyi tanımla """
    gecici_dosya = "gecici_ses.wav"
    ses_kaydet(gecici_dosya)

    ozellikler = ozellikleri_cikar(gecici_dosya)
    if ozellikler is None:
        print("❌ Kaydedilen ses özellikleri çıkarılamadı!")
        return

    if hasattr(model, "predict"):
        try:
            ogrenci_no = model.predict([ozellikler])[0]
            ogrenci_adi = ogrenci_verileri.get(ogrenci_no, "Bilinmeyen")
            print(f"✅ {ogrenci_adi} ({ogrenci_no}) derste!")
        except Exception as e:
            print(f"❌ Ses tanınamadı: {e}")
    else:
        print("❌ Model henüz eğitilmedi!")

def main():
    while True:
        print("\n📌 Bir işlem seçin:")
        print("1️⃣ Yeni Öğrenci Kaydı")
        print("2️⃣ Yoklama Al")
        print("3️⃣ Öğrenci Sil")
        print("4️⃣ Çıkış")
        secim = input("👉 Seçiminizi girin: ").strip()

        if secim == "1":
            ogrenci_kayit()
        elif secim == "2":
            yoklama_al()
        elif secim == "3":
            ogrenci_sil()
        elif secim == "4":
            print("👋 Güle güle!")
            break
        else:
            print("❌ Geçersiz seçim, tekrar deneyin!")

if __name__ == "__main__":
    main()