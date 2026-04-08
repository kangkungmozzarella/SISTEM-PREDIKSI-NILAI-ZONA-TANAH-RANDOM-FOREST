
# SETUP INSTRUCTIONS - PREDIKSI NILAI ZONA TANAH IDI RAYEUK

## 1. INSTALASI DEPENDENCIES
```bash
pip install -r flask_components/requirements.txt
```

## 2. STRUKTUR PROJECT FLASK
```
your_flask_project/
├── app.py                          # Main Flask application
├── templates/                      # HTML templates
│   ├── index.html                 # Home page
│   ├── predict.html               # Prediction form
│   └── dashboard.html             # Analytics dashboard
├── static/                        # CSS, JS, images
│   ├── css/
│   ├── js/
│   └── maps/                      # Leaflet map files
└── flask_components/              # ML components (copy dari Jupyter)
    ├── models/                    # Trained models
    ├── data/                      # Datasets
    ├── results/                   # Analysis results
    └── visualizations/            # Interactive charts
```

## 3. SAMPLE FLASK APP CODE

### app.py
```python
from flask import Flask, render_template, request, jsonify
import pickle
import json
import pandas as pd

app = Flask(__name__)

# Load ML pipeline
with open('flask_components/models/complete_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

with open('flask_components/models/prediction_functions.pkl', 'rb') as f:
    prediction_functions = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        input_data = {
            'koordinat_latitude': float(request.form['latitude']),
            'koordinat_longitude': float(request.form['longitude']),
            'luas_zona_m2': float(request.form['luas_zona']),
            'jarak_pusat_kota_km': float(request.form['jarak_pusat']),
            'elevasi_mdpl': float(request.form['elevasi']),
            'kemiringan_lereng_persen': float(request.form['kemiringan']),
            'kepadatan_penduduk_km2': float(request.form['kepadatan']),
            'jarak_jalan_utama_m': float(request.form['jarak_jalan']),
            'jarak_sekolah_m': float(request.form['jarak_sekolah']),
            'jarak_puskesmas_m': float(request.form['jarak_puskesmas']),
            'jarak_pasar_m': float(request.form['jarak_pasar']),
            'status_listrik': int(request.form['status_listrik']),
            'status_air_bersih': int(request.form['status_air']),
            'aksesibilitas_skor': float(request.form['aksesibilitas']),
            'tahun_data': int(request.form['tahun']),
            'nama_kelurahan': request.form['kelurahan'],
            'jenis_penggunaan_lahan': request.form['jenis_lahan'],
            'nomor_zona': int(request.form.get('nomor_zona', 999))
        }

        # Predict
        result = prediction_functions['predict_land_value'](input_data, pipeline)

        return jsonify(result)

    return render_template('predict.html')

@app.route('/dashboard')
def dashboard():
    # Load analytics data
    with open('flask_components/results/final_performance_summary.json', 'r') as f:
        performance = json.load(f)

    return render_template('dashboard.html', performance=performance)

@app.route('/api/zones_data')
def zones_data():
    # Return zone data for Leaflet map
    df = pd.read_csv('flask_components/data/sorted_dataset.csv')
    zones_data = df[['id_zona', 'nama_kelurahan', 'koordinat_latitude', 
                     'koordinat_longitude', 'nilai_tanah_per_m2', 'tahun_data']].to_dict('records')
    return jsonify(zones_data)

if __name__ == '__main__':
    app.run(debug=True)
```

## 4. LEAFLET INTEGRATION
Untuk integrasi Leaflet maps, load zone data dan tampilkan pada peta interaktif:
```javascript
// Load zone data
fetch('/api/zones_data')
    .then(response => response.json())
    .then(data => {
        data.forEach(zone => {
            L.marker([zone.koordinat_latitude, zone.koordinat_longitude])
             .addTo(map)
             .bindPopup(`
                 <b>${zone.nama_kelurahan}</b><br>
                 Nilai: Rp ${zone.nilai_tanah_per_m2.toLocaleString()}/m²<br>
                 Tahun: ${zone.tahun_data}
             `);
        });
    });
```

## 5. DEPLOYMENT
Untuk deployment ke production, gunakan:
```bash
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

## 6. FITUR YANG TERSEDIA
- ✅ Prediksi nilai tanah real-time
- ✅ Interactive maps dengan Leaflet  
- ✅ Analytics dashboard dengan Plotly
- ✅ Model performance metrics
- ✅ Error analysis per kelurahan/tahun
- ✅ Feature importance visualization
- ✅ RESTful API endpoints

