from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import sqlite3
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'supersifre'  # Flash mesajlar ve session için gizli anahtar

# Fotoğrafların yükleneceği klasör
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# İzin verilen resim uzantıları
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Dosya uzantısı kontrol fonksiyonu
def izin_verilen_dosya(dosya_adi):
    return '.' in dosya_adi and dosya_adi.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Ana sayfa: ders ve öğrenci listesini alır ve index.html'e yollar
@app.route('/')
def ana_sayfa():
    conn = sqlite3.connect('school.db')
    cursor = conn.cursor()
    
    # Dersleri getir
    cursor.execute('SELECT ders_id, ders_adi FROM dersler')
    dersler = cursor.fetchall()
    
    # Öğrencileri ve ders adlarını getir (JOIN)
    cursor.execute('''
        SELECT o.ogrenci_id, o.tam_adi, o.ogrenci_numarasi, o.foto_yolu, o.ders_id, d.ders_adi
        FROM ogrenciler o
        JOIN dersler d ON o.ders_id = d.ders_id
    ''')
    ogrenciler = cursor.fetchall()
    
    conn.close()
    return render_template('index.html', dersler=dersler, ogrenciler=ogrenciler)

# Ders ekleme işlemi
@app.route('/ders_ekle', methods=['POST'])
def ders_ekle():
    ders_adi = request.form['ders_adi']
    if ders_adi:
        conn = sqlite3.connect('school.db')
        cursor = conn.cursor()
        cursor.execute('INSERT INTO dersler (ders_adi) VALUES (?)', (ders_adi,))
        conn.commit()
        conn.close()
        flash("Ders başarıyla eklendi", "success")
    else:
        flash("Lütfen ders adını girin", "error")
    return redirect(url_for('ana_sayfa'))

# Yeni öğrenci ekleme
@app.route('/ogrenci_ekle', methods=['POST'])
def ogrenci_ekle():
    tam_adi = request.form['tam_adi']
    ogrenci_numarasi = request.form['ogrenci_numarasi']
    ders_id = request.form['ders_id']
    
    # Fotoğraf kontrolü
    if 'foto' not in request.files:
        flash("Fotoğraf yüklenmedi", "error")
        return redirect(url_for('ana_sayfa'))
    
    foto = request.files['foto']
    if foto.filename == '':
        flash("Fotoğraf seçilmedi", "error")
        return redirect(url_for('ana_sayfa'))
    
    if foto and izin_verilen_dosya(foto.filename):
        dosya_adi = secure_filename(foto.filename)
        foto_yolu = os.path.join(app.config['UPLOAD_FOLDER'], dosya_adi)
        foto.save(foto_yolu)
        
        # Öğrenciyi veritabanına kaydet
        conn = sqlite3.connect('school.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO ogrenciler (tam_adi, ogrenci_numarasi, foto_yolu, ders_id)
            VALUES (?, ?, ?, ?)
        ''', (tam_adi, ogrenci_numarasi, foto_yolu, ders_id))
        conn.commit()
        conn.close()
        flash("Öğrenci başarıyla eklendi", "success")
    else:
        flash("Geçersiz dosya uzantısı", "error")
    
    return redirect(url_for('ana_sayfa'))

# Öğrenciyi tamamen sil (tüm derslerinden)
@app.route('/ogrenci_sil/<int:ogrenci_id>')
def ogrenci_sil(ogrenci_id):
    conn = sqlite3.connect('school.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM ogrenciler WHERE ogrenci_id = ?', (ogrenci_id,))
    cursor.execute('DELETE FROM yoklama WHERE ogrenci_id = ?', (ogrenci_id,))
    conn.commit()
    conn.close()
    flash("Öğrenci başarıyla silindi", "success")
    return redirect(url_for('ana_sayfa'))

# Öğrenciyi sadece belli dersten sil
@app.route('/ogrenciyi_dersten_sil/<int:ogrenci_id>/<int:ders_id>')
def ogrenciyi_dersten_sil(ogrenci_id, ders_id):
    conn = sqlite3.connect('school.db')
    cursor = conn.cursor()
    
    # Sadece o dersten sil
    cursor.execute('DELETE FROM ogrenciler WHERE ogrenci_id = ? AND ders_id = ?', (ogrenci_id, ders_id))
    cursor.execute('DELETE FROM yoklama WHERE ogrenci_id = ? AND ders_id = ?', (ogrenci_id, ders_id))
    
    conn.commit()
    conn.close()
    flash("Öğrenci dersten başarıyla silindi", "success")
    return redirect(url_for('ana_sayfa'))

# Uygulama çalıştırıldığında klasör oluştur ve başlat
if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
