# Implementasi SVM dan Naive Bayes untuk Prediksi ISPA

## Deskripsi Proyek
Proyek ini mengimplementasikan algoritma Support Vector Machine (SVM) dan Naive Bayes untuk memprediksi tingkat keparahan penyakit Infeksi Saluran Pernapasan Akut (ISPA) berdasarkan data dari RSUD Fauziah Bireuen.

## Struktur File
- `flask_app/` - Folder aplikasi Flask
  - `static/` - Folder untuk file statis
    - `data/` - Data hasil analisis
    - `models/` - Model machine learning
    - `images/` - Visualisasi dan grafik
    - `js/` - File JavaScript
    - `css/` - File CSS
  - `templates/` - Template HTML

## Dataset
Dataset berisi 831 pasien dari Poli Paru RSUD Fauziah Bireuen dengan 17 kasus ISPA (Pneumonia).

## Fitur
Fitur yang digunakan dalam model:
- Umur Pasien
- Jenis Kelamin
- Status Daftar (Baru/Lama)

## Performa Model
- SVM: Akurasi 1.0000
- Naive Bayes: Akurasi 1.0000
- Model terbaik: Tie

## Cara Menjalankan Aplikasi
1. Pastikan semua dependensi terinstall
```
pip install -r requirements.txt
```

2. Jalankan aplikasi Flask
```
cd flask_app
python app.py
```

3. Buka browser dan akses `http://localhost:5000`

## Kontributor
- Nama Anda
- Email Anda

## Tanggal Dibuat
04 November 2025
