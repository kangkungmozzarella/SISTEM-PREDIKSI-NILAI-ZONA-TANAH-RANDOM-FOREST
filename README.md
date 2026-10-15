# Sistem Prediksi Nilai Zona Tanah — Random Forest + WebGIS

Prediksi nilai zona tanah (Rp/m²) di **Kecamatan Idi Rayeuk, Kabupaten Aceh Timur** menggunakan
Random Forest Regression, dengan peta interaktif berbasis Leaflet.js.

## Performa model (data testing)

| Metrik | Nilai |
|---|---|
| R² | 0.9864 |
| MAPE | 4.95% |
| MAE | Rp 9.774 /m² |
| RMSE | Rp 13.761 /m² |

Dataset: 452 baris (2023–2025), 43 gampong/kelurahan, 3 jenis lahan (Komersial, Pemukiman, Pertanian).
Split acak 80/20 (361 training, 91 testing, `random_state=42`).

Fitur paling berpengaruh: jarak ke pasar (33,2%), jarak ke puskesmas (28,0%), luas zona (14,1%),
kepadatan penduduk (13,4%).

## Struktur

```
app.py                  Aplikasi Flask (route halaman, dataset, API)
prediction.py           Memuat model & melakukan prediksi; membaca hasil evaluasi
latih.ipynb             Notebook training (EDA → preprocessing → training → evaluasi → export)
db/random-forest.sql    Skema + data MySQL (tabel tbl_dataset)
dataset.csv             Dataset mentah
flask_components/
  models/               complete_pipeline.pkl (model + label encoder), dll.
  data/                 Dataset terproses dan detail split
  results/              Metrik, feature importance, prediksi vs aktual
  visualizations/       Grafik Plotly (HTML) yang ditampilkan di halaman analisis
templates/, static/     Tampilan
tes.py                  Utilitas: ekstrak batas desa Idi Rayeuk dari GeoJSON GADM
```

## Menjalankan

1. Buat virtual environment dan pasang dependensi:
   ```bash
   python -m venv venv
   venv\Scripts\activate          # Windows
   source venv/bin/activate       # macOS/Linux
   pip install -r requirements.txt
   ```
   `scikit-learn` harus versi **1.8.0**, sama dengan versi saat model disimpan. Versi lain bisa
   gagal memuat `complete_pipeline.pkl` atau memberi hasil berbeda.

2. Impor database (opsional, tetapi dibutuhkan untuk halaman Dataset, Dashboard, dan peta):
   ```bash
   mysql -u root -e "CREATE DATABASE \`random-forest\`"
   mysql -u root random-forest < db/random-forest.sql
   ```

3. Jalankan aplikasi:
   ```bash
   python app.py
   ```
   Buka http://localhost:5000. Login admin di `/admin/login`.

Prediksi (`/api/predict`) dan halaman analisis tetap berjalan walaupun MySQL tidak tersedia,
karena keduanya membaca file di `flask_components/`.

### Konfigurasi (environment variable)

| Variabel | Default | Keterangan |
|---|---|---|
| `SECRET_KEY` | nilai bawaan di `app.py` | **Wajib diganti** saat deploy |
| `ADMIN_USERNAME` / `ADMIN_PASSWORD` | `admin` / `admin123` | **Wajib diganti** saat deploy |
| `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME` | `localhost`, `3306`, `root`, kosong, `random-forest` | Koneksi MySQL |
| `FLASK_DEBUG` | `0` | Set `1` untuk mode debug |
| `HOST` / `PORT` | `127.0.0.1` / `5000` | Set `HOST=0.0.0.0` agar bisa diakses dari jaringan |

## API prediksi

`POST /api/predict` menerima JSON atau form data dengan nama kolom sesuai dataset.
`nama_kelurahan` dan `jenis_penggunaan_lahan` wajib diisi; fitur numerik yang tidak dikirim
diisi median zona di kelurahan (dan jenis lahan) yang sama.

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"nama_kelurahan": "Keude Blang", "jenis_penggunaan_lahan": "Komersial",
       "luas_zona_m2": 500, "jarak_pusat_kota_km": 0.5, "aksesibilitas_skor": 9,
       "jarak_pasar_m": 300, "tahun_data": 2025}'
```

Respons berisi `prediction` (Rp/m²), `range` (± MAE), `category` (Nilai Rendah/Sedang/Tinggi,
berdasarkan tertil dataset), `input_used`, dan `filled_defaults` (fitur yang diisi otomatis).

## Melatih ulang model

Jalankan semua sel di `latih.ipynb`. Notebook menulis ulang isi `flask_components/`. Jika versi
scikit-learn berubah, perbarui juga versi di `requirements.txt`.
