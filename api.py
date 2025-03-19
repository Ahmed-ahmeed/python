import os
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import pickle
from sklearn.neighbors import KNeighborsClassifier
from flask import Flask, request, jsonify

# 📂 Kayıtların saklanacağı klasör ve model dosyası
VERI_KLASORU = "dataset"
MODEL_DOSYA = "voice_model.pkl"

if not os.path.exists(VERI_KLASORU):
    os.makedirs(VERI_KLASORU)

# 🎤 Ses kayıt ayarları
SURE = 1  # Kayıt süresi (saniye)
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
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfccs, axis=1)
    except Exception as e:
        print(f"❌ Ses özellikleri çıkarılamadı: {e}")
        return None

# تعريف الـ Flask قبل استخدامه
app = Flask(__name__)

@app.route('/kayit', methods=['POST'])
def ogrenci_kayit():
    """ 📝 Yeni öğrenci kaydı al """
    data = request.get_json()
    isim = data['isim']
    ogrenci_no = data['ogrenci_no']

    dosya_yolu = os.path.join(VERI_KLASORU, f"{ogrenci_no}.wav")
    ses_kaydet(dosya_yolu)

    ozellikler = ozellikleri_cikar(dosya_yolu)
    if ozellikler is None:
        return jsonify({"status": "error", "message": "Özellik çıkarma başarısız!"})

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

        return jsonify({"status": "success", "message": f"{isim} ({ogrenci_no}) başarıyla kaydedildi!"})
    else:
        return jsonify({"status": "error", "message": "Model güncellenemedi!"})

@app.route('/yoklama', methods=['POST'])
def yoklama_al():
    """ 📋 Yoklama al ve öğrenciyi tanımla """
    data = request.get_json()
    gecici_dosya = "gecici_ses.wav"
    ses_kaydet(gecici_dosya)

    ozellikler = ozellikleri_cikar(gecici_dosya)
    if ozellikler is None:
        return jsonify({"status": "error", "message": "Kaydedilen ses özellikleri çıkarılamadı!"})

    if hasattr(model, "predict"):
        try:
            ogrenci_no = model.predict([ozellikler])[0]
            ogrenci_adi = ogrenci_verileri.get(ogrenci_no, "Bilinmeyen")
            return jsonify({"status": "success", "message": f"{ogrenci_adi} ({ogrenci_no}) derste!"})
        except Exception as e:
            return jsonify({"status": "error", "message": f"Ses tanınamadı: {e}"})
    else:
        return jsonify({"status": "error", "message": "Model henüz eğitilmedi!"})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
