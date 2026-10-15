from flask import Flask, redirect, request, render_template, url_for, session, flash, jsonify, send_from_directory
import pandas as pd
import os
import hmac
from functools import wraps
from datetime import datetime
import pymysql
import csv
import io

import prediction as model_service

# Warnings
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VISUALIZATION_DIR = os.path.join(BASE_DIR, 'flask_components', 'visualizations')

app = Flask(__name__)

# Konfigurasi bisa di-override lewat environment variable (lihat README)
app.secret_key = os.environ.get('SECRET_KEY', 'land-value-prediction-secret-key-2025')
ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'admin123')

# Database Configuration
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'user': os.environ.get('DB_USER', 'root'),
    'password': os.environ.get('DB_PASSWORD', ''),
    'database': os.environ.get('DB_NAME', 'random-forest'),
    'port': int(os.environ.get('DB_PORT', 3306)),
    'charset': 'utf8mb4',
    'autocommit': True
}

# Database connection function
def get_db_connection():
    """Membuat koneksi ke database MySQL"""
    try:
        connection = pymysql.connect(**DB_CONFIG)
        return connection
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('Silakan login untuk mengakses halaman ini', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Helper function untuk load data dari database
def load_land_value_data():
    """Load dataset nilai zona tanah dari database"""
    try:
        connection = get_db_connection()
        if connection:
            df = pd.read_sql("""
                SELECT id_zona, nama_kelurahan, koordinat_latitude, koordinat_longitude,
                       nomor_zona, luas_zona_m2, nilai_tanah_per_m2, jenis_penggunaan_lahan,
                       jarak_pusat_kota_km, elevasi_mdpl, kemiringan_lereng_persen,
                       kepadatan_penduduk_km2, jarak_jalan_utama_m, jarak_sekolah_m,
                       jarak_puskesmas_m, jarak_pasar_m, status_listrik, status_air_bersih,
                       aksesibilitas_skor, tahun_data
                FROM tbl_dataset 
                ORDER BY tahun_data DESC, nama_kelurahan ASC
            """, connection)
            connection.close()
            return df
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Nama file visualisasi dari notebook untuk tiap panel di halaman analisis
VISUALIZATION_FILES = {
    'feature_importance_chart': 'feature_importance_bar.html',
    'feature_category_chart': 'feature_importance_category.html',
    'prediction_scatter': 'prediksi_vs_aktual_scatter.html',
    'residual_plot': 'residual_plot.html',
    'error_distribution': 'error_distribution.html',
    'metrics_chart': 'metrics_comparison_train_test.html',
}

def load_visualizations():
    """Daftar visualisasi HTML yang tersedia di flask_components/visualizations"""
    return {
        key: filename
        for key, filename in VISUALIZATION_FILES.items()
        if os.path.exists(os.path.join(VISUALIZATION_DIR, filename))
    }

def get_model_mape():
    """MAPE testing set dari hasil notebook (persen)"""
    try:
        return model_service.get_model_metrics()['mape']
    except Exception as e:
        print(f"Error loading model metrics: {e}")
        return None

def generate_sample_stats():
    """Generate sample statistics untuk development"""
    return {
        'total_zones': 452,
        'total_kelurahan': 43,
        'total_years': 3,
        'data_period': '2023-2025',
        'model_accuracy': get_model_mape(),
        'features_count': 20,
        'price_range': {
            'min': 68000,
            'max': 520000,
            'avg': 245000
        },
        'land_types': {
            'Komersial': 158,
            'Pemukiman': 204, 
            'Pertanian': 91
        },
        'kelurahan_coverage': [
            'Idi Rayeuk', 'Peukan Bada', 'Simpang Jernih', 'Idi Tunong',
            'Banda Alam', 'Gampong Jawa', 'Keude Panggoi', 'Julok'
        ]
    }

@app.route('/')
def homepage():
    """Homepage publik untuk prediksi nilai zona tanah"""
    try:
        stats = generate_sample_stats()
        try:
            kelurahan_list = model_service.get_kelurahan_list()
            model_status = 'Active'
        except Exception as e:
            print(f"Error loading model: {e}")
            kelurahan_list = []
            model_status = 'Model tidak tersedia'

        homepage_data = {
            'total_zones': stats['total_zones'],
            'total_kelurahan': stats['total_kelurahan'],
            'total_years': stats['total_years'],
            'data_period': stats['data_period'],
            'features_count': stats['features_count'],
            'model_accuracy': stats['model_accuracy'],
            'price_min': stats['price_range']['min'],
            'price_max': stats['price_range']['max'],
            'price_avg': stats['price_range']['avg'],
            'research_title': 'Prediksi Nilai Zona Tanah di Kecamatan Idi Rayeuk, Aceh Timur',
            'research_location': 'Kecamatan Idi Rayeuk, Aceh Timur',
            'research_method': 'Random Forest Regression berbasis GIS',
            'main_kecamatan': 'Idi Rayeuk',
            'algorithm': 'Random Forest',
            'platform': 'Flask + Leaflet.js',
            'kelurahan_list': kelurahan_list,
            'model_status': model_status,
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        return render_template('index.html', **homepage_data)

    except Exception as e:
        print(f"Homepage error: {e}")
        fallback_data = {
            'total_zones': 452,
            'total_kelurahan': 43,
            'total_years': 3,
            'features_count': 20,
            'model_accuracy': None,
            'kelurahan_list': [],
            'research_title': 'Prediksi Nilai Zona Tanah Idi Rayeuk',
            'algorithm': 'Random Forest'
        }
        return render_template('index.html', **fallback_data)

#-------Login---------#
@app.route('/admin/login', methods=['GET', 'POST'])
def login():
    """Login admin untuk dashboard"""
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')

        if (hmac.compare_digest(username, ADMIN_USERNAME)
                and hmac.compare_digest(password, ADMIN_PASSWORD)):
            session['username'] = username
            session['status'] = "Login"
            flash(f"Selamat datang di Sistem Prediksi Nilai Zona Tanah, {username}!", 'success')
            return redirect(url_for('dashboard'))
        else:
            flash("Username atau Password Salah", 'danger')

    return render_template('admin/login.html')

@app.route('/logout')
def logout():
    """Logout admin"""
    session.clear()
    flash('Anda telah logout', 'info')
    return redirect(url_for('homepage'))

#-------Dashboard---------#
@app.route('/dashboard')
@login_required
def dashboard():
    """Dashboard utama untuk prediksi nilai zona tanah"""
    try:
        land_data = load_land_value_data()

        if land_data is not None and len(land_data) > 0:
            dashboard_data = {
                'total_zones': len(land_data),
                'total_kelurahan': land_data['nama_kelurahan'].nunique(),
                'total_years': land_data['tahun_data'].nunique(),
                'data_period': f"{land_data['tahun_data'].min()}-{land_data['tahun_data'].max()}",
                'features_count': 20,
                'price_min': int(land_data['nilai_tanah_per_m2'].min()),
                'price_max': int(land_data['nilai_tanah_per_m2'].max()),
                'price_avg': int(land_data['nilai_tanah_per_m2'].mean()),
                'land_types': land_data['jenis_penggunaan_lahan'].value_counts().to_dict(),
                'komersial_count': len(land_data[land_data['jenis_penggunaan_lahan'] == 'Komersial']),
                'pemukiman_count': len(land_data[land_data['jenis_penggunaan_lahan'] == 'Pemukiman']),
                'pertanian_count': len(land_data[land_data['jenis_penggunaan_lahan'] == 'Pertanian']),
                'kelurahan_list': sorted(land_data['nama_kelurahan'].unique().tolist()),
                'main_kecamatan': 'Idi Rayeuk',
                'kabupaten': 'Aceh Timur',
                'provinsi': 'Aceh',
                'algorithm': 'Random Forest Regression',
                'platform': 'WebGIS + Leaflet JS',
                'model_accuracy': get_model_mape(),
                'evaluation_metrics': ['MAE', 'MAPE', 'RMSE', 'R²'],
                'research_title': 'Prediksi Nilai Zona Tanah di Kecamatan Idi Rayeuk Aceh Timur menggunakan Random Forest Regression berbasis Sistem Informasi Geografis',
                'research_location': 'Kecamatan Idi Rayeuk, Kabupaten Aceh Timur',
                'research_period': f"{land_data['tahun_data'].min()}-{land_data['tahun_data'].max()}",
                'research_method': 'Random Forest Regression + GIS',
                'data_status': 'Real Data from Database',
                'model_status': 'Active',
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        else:
            stats = generate_sample_stats()
            dashboard_data = {
                'total_zones': stats['total_zones'],
                'total_kelurahan': stats['total_kelurahan'], 
                'total_years': stats['total_years'],
                'data_period': stats['data_period'],
                'model_accuracy': stats['model_accuracy'],
                'features_count': stats['features_count'],
                'price_min': stats['price_range']['min'],
                'price_max': stats['price_range']['max'],
                'price_avg': stats['price_range']['avg'],
                'land_types': stats['land_types'],
                'komersial_count': stats['land_types']['Komersial'],
                'pemukiman_count': stats['land_types']['Pemukiman'],
                'pertanian_count': stats['land_types']['Pertanian'],
                'kelurahan_list': stats['kelurahan_coverage'],
                'main_kecamatan': 'Idi Rayeuk',
                'kabupaten': 'Aceh Timur',
                'provinsi': 'Aceh',
                'algorithm': 'Random Forest Regression',
                'platform': 'WebGIS + Leaflet JS',
                'evaluation_metrics': ['MAE', 'MAPE', 'RMSE', 'R²'],
                'research_title': 'Prediksi Nilai Zona Tanah di Kecamatan Idi Rayeuk Aceh Timur menggunakan Random Forest Regression berbasis Sistem Informasi Geografis',
                'research_location': 'Kecamatan Idi Rayeuk, Kabupaten Aceh Timur',
                'research_period': '2023-2025',
                'research_method': 'Random Forest Regression + GIS',
                'data_status': 'No Database Connection - Using Sample Data',
                'model_status': 'Development',
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
        return render_template('dashboard/index.html', **dashboard_data)
        
    except Exception as e:
        flash(f'Error loading dashboard data: {str(e)}', 'danger')
        print(f"Dashboard error: {str(e)}")
        minimal_data = {
            'total_zones': 452,
            'total_kelurahan': 43,
            'model_accuracy': get_model_mape(),
            'data_period': '2023-2025',
            'research_title': 'Prediksi Nilai Zona Tanah Idi Rayeuk',
            'research_location': 'Kecamatan Idi Rayeuk, Aceh Timur',
            'data_status': 'Error - Using Minimal Data',
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        return render_template('dashboard/index.html', **minimal_data)

#-------Dataset Management Routes---------#
@app.route('/dataset')
@login_required
def dataset_index():
    """Halaman utama untuk mengelola dataset zona tanah"""
    try:
        land_data = load_land_value_data()
        
        if land_data is not None and len(land_data) > 0:
            land_types_count = land_data['jenis_penggunaan_lahan'].value_counts()
            
            stats = {
                'total_zones': len(land_data),
                'total_kelurahan': land_data['nama_kelurahan'].nunique(),
                'years': sorted(land_data['tahun_data'].unique().tolist()),
                'land_types': land_types_count.to_dict(),
                'price_stats': {
                    'min': int(land_data['nilai_tanah_per_m2'].min()),
                    'max': int(land_data['nilai_tanah_per_m2'].max()),
                    'mean': int(land_data['nilai_tanah_per_m2'].mean()),
                    'median': int(land_data['nilai_tanah_per_m2'].median())
                }
            }
            dataset_list = land_data.to_dict('records')
        else:
            stats = {
                'total_zones': 0,
                'total_kelurahan': 0,
                'years': [],
                'land_types': {'Komersial': 0, 'Pemukiman': 0, 'Pertanian': 0},
                'price_stats': {'min': 0, 'max': 0, 'mean': 0, 'median': 0}
            }
            dataset_list = []
            flash('Tidak ada data di database', 'warning')
        
        return render_template('dataset/index.html', 
                             dataset=dataset_list, 
                             stats=stats,
                             total_records=len(dataset_list))
        
    except Exception as e:
        flash(f'Error loading dataset: {str(e)}', 'danger')
        print(f"Dataset error: {str(e)}")
        return render_template('dataset/index.html', 
                             dataset=[], 
                             stats={
                                 'total_zones': 0,
                                 'total_kelurahan': 0,
                                 'years': [],
                                 'land_types': {'Komersial': 0, 'Pemukiman': 0, 'Pertanian': 0},
                                 'price_stats': {'min': 0, 'max': 0, 'mean': 0, 'median': 0}
                             },
                             total_records=0)

@app.route('/dataset/upload', methods=['POST'])
@login_required
def dataset_upload():
    """Upload dataset CSV"""
    try:
        if 'file' not in request.files:
            flash('Tidak ada file yang dipilih', 'danger')
            return redirect(url_for('dataset_index'))
        
        file = request.files['file']
        if file.filename == '':
            flash('Tidak ada file yang dipilih', 'danger')
            return redirect(url_for('dataset_index'))
        
        if file and file.filename.lower().endswith('.csv'):
            stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
            csv_input = csv.DictReader(stream)
            
            connection = get_db_connection()
            if not connection:
                flash('Error koneksi database', 'danger')
                return redirect(url_for('dataset_index'))
            
            cursor = connection.cursor()
            
            inserted_count = 0
            for row in csv_input:
                try:
                    insert_query = """
                    INSERT INTO tbl_dataset (
                        id_zona, nama_kelurahan, koordinat_latitude, koordinat_longitude,
                        nomor_zona, luas_zona_m2, nilai_tanah_per_m2, jenis_penggunaan_lahan,
                        jarak_pusat_kota_km, elevasi_mdpl, kemiringan_lereng_persen,
                        kepadatan_penduduk_km2, jarak_jalan_utama_m, jarak_sekolah_m,
                        jarak_puskesmas_m, jarak_pasar_m, status_listrik, status_air_bersih,
                        aksesibilitas_skor, tahun_data
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    nilai_tanah_per_m2 = VALUES(nilai_tanah_per_m2),
                    updated_at = CURRENT_TIMESTAMP
                    """
                    
                    cursor.execute(insert_query, (
                        row['id_zona'], row['nama_kelurahan'], 
                        float(row['koordinat_latitude']), float(row['koordinat_longitude']),
                        int(row['nomor_zona']), int(row['luas_zona_m2']), 
                        int(row['nilai_tanah_per_m2']), row['jenis_penggunaan_lahan'],
                        float(row['jarak_pusat_kota_km']), int(row['elevasi_mdpl']),
                        float(row['kemiringan_lereng_persen']), int(row['kepadatan_penduduk_km2']),
                        int(row['jarak_jalan_utama_m']), int(row['jarak_sekolah_m']),
                        int(row['jarak_puskesmas_m']), int(row['jarak_pasar_m']),
                        int(row['status_listrik']), int(row['status_air_bersih']),
                        float(row['aksesibilitas_skor']), int(row['tahun_data'])
                    ))
                    inserted_count += 1
                except Exception as row_error:
                    print(f"Error inserting row: {row_error}")
                    continue
            
            connection.commit()
            cursor.close()
            connection.close()
            
            flash(f'Berhasil upload {inserted_count} data zona tanah', 'success')
        else:
            flash('File harus berformat CSV', 'danger')
            
    except Exception as e:
        flash(f'Error upload file: {str(e)}', 'danger')
        print(f"Upload error: {str(e)}")
    
    return redirect(url_for('dataset_index'))

@app.route('/dataset/reset', methods=['POST'])
@login_required
def dataset_reset():
    """Reset semua data dataset"""
    try:
        connection = get_db_connection()
        if connection:
            cursor = connection.cursor()
            cursor.execute("DELETE FROM tbl_dataset")
            connection.commit()
            cursor.close()
            connection.close()
            flash('Semua data berhasil direset', 'success')
        else:
            flash('Error koneksi database', 'danger')
    except Exception as e:
        flash(f'Error reset data: {str(e)}', 'danger')
    
    return redirect(url_for('dataset_index'))

# =====================================================================
# API ZONES - DIUPDATE: return semua zona dengan koordinat unik per zona
# =====================================================================
@app.route('/api/dataset/zones')
def api_zones_data():
    """
    API endpoint untuk mendapatkan data semua zona (untuk peta di dataset page).
    Mengembalikan semua zona dengan koordinat unik per zona (bukan per kelurahan),
    sehingga tiap bidang tanah tampil sebagai marker terpisah di peta.
    """
    try:
        land_data = load_land_value_data()
        
        if land_data is not None:
            # Ambil semua zona (semua tahun), bukan hanya tahun terbaru
            # Karena tiap zona sudah punya koordinat unik, semua bisa ditampilkan
            # Group by koordinat unik (1 marker per bidang tanah, bukan per baris tahun)
            # Gunakan tahun terbaru per zona untuk info popup
            
            # Identifikasi zona unik berdasarkan kombinasi koordinat + kelurahan
            land_data['zona_key'] = (
                land_data['koordinat_latitude'].astype(str) + '_' +
                land_data['koordinat_longitude'].astype(str) + '_' +
                land_data['nama_kelurahan']
            )
            
            zones_data = []
            
            # Per zona unik, ambil data tahun terbaru untuk popup
            # Tapi sertakan juga daftar semua tahun yang tersedia
            for zona_key, group in land_data.groupby('zona_key'):
                # Data tahun terbaru untuk info utama
                latest = group.sort_values('tahun_data', ascending=False).iloc[0]
                
                # Semua data tahun untuk ditampilkan di popup
                tahun_list = sorted(group['tahun_data'].unique().tolist())
                nilai_per_tahun = {}
                for _, row in group.iterrows():
                    nilai_per_tahun[int(row['tahun_data'])] = int(row['nilai_tanah_per_m2'])
                
                zones_data.append({
                    'id_zona': latest['id_zona'],
                    'nama_kelurahan': latest['nama_kelurahan'],
                    'lat': float(latest['koordinat_latitude']),
                    'lng': float(latest['koordinat_longitude']),
                    'nomor_zona': int(latest['nomor_zona']),
                    'luas_zona_m2': int(latest['luas_zona_m2']),
                    'nilai_tanah_per_m2': int(latest['nilai_tanah_per_m2']),
                    'jenis_penggunaan_lahan': latest['jenis_penggunaan_lahan'],
                    'jarak_pusat_kota_km': float(latest['jarak_pusat_kota_km']),
                    'aksesibilitas_skor': float(latest['aksesibilitas_skor']),
                    'status_listrik': int(latest['status_listrik']),
                    'status_air_bersih': int(latest['status_air_bersih']),
                    'tahun_terbaru': int(latest['tahun_data']),
                    'tahun_list': tahun_list,
                    'nilai_per_tahun': nilai_per_tahun,
                    'id_zona_list': sorted(group['id_zona'].unique().tolist())
                })
            
            return jsonify({
                'status': 'success',
                'total': len(zones_data),
                'data': zones_data
            })
        else:
            return jsonify({'status': 'error', 'total': 0, 'data': []})
            
    except Exception as e:
        print(f"API zones data error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e), 'total': 0, 'data': []})

@app.route('/tentang')
def tentang():
    """Halaman tentang penelitian"""
    try:
        land_data = load_land_value_data()
        
        if land_data is not None and len(land_data) > 0:
            total_data = len(land_data)
            total_kelurahan = land_data['nama_kelurahan'].nunique()
            data_period = f"{land_data['tahun_data'].min()}-{land_data['tahun_data'].max()}"
            price_range = {
                'min': int(land_data['nilai_tanah_per_m2'].min()),
                'max': int(land_data['nilai_tanah_per_m2'].max()),
                'avg': int(land_data['nilai_tanah_per_m2'].mean())
            }
        else:
            total_data = 452
            total_kelurahan = 43
            data_period = "2023-2025"
            price_range = {'min': 68000, 'max': 520000, 'avg': 245000}
        
        tentang_data = {
            'research_title': 'Prediksi Nilai Zona Tanah di Kecamatan Idi Rayeuk Aceh Timur Menggunakan Metode Random Forest Regression Berbasis Sistem Informasi Geografis',
            'research_location': 'Kecamatan Idi Rayeuk, Kabupaten Aceh Timur, Provinsi Aceh',
            'year': '2025',
            'total_zones': total_data,
            'total_kelurahan': total_kelurahan,
            'data_period': data_period,
            'price_range': price_range,
            'features_count': 20,
            'algorithm': 'Random Forest Regression',
            'model_performance': {
                'mape': '4.95%',
                'r2': '0.9864',
                'status': 'Outstanding'
            },
            'main_objective': 'Mengembangkan sistem prediksi nilai zona tanah yang akurat menggunakan machine learning dan GIS untuk mendukung perencanaan tata ruang dan penilaian properti',
            'specific_objectives': [
                'Menganalisis faktor-faktor yang mempengaruhi nilai zona tanah',
                'Mengimplementasikan algoritma Random Forest untuk prediksi nilai tanah',
                'Mengintegrasikan sistem prediksi dengan platform WebGIS',
                'Mengevaluasi akurasi model prediksi yang dikembangkan',
                'Memberikan rekomendasi untuk perencanaan tata ruang wilayah'
            ],
            'benefits': [
                'Pemerintah Daerah: Perencanaan tata ruang yang lebih baik',
                'Penilai Properti: Estimasi nilai tanah yang objektif dan akurat',
                'Investor: Analisis investasi properti yang lebih informed',
                'Akademisi: Referensi untuk penelitian selanjutnya',
                'Masyarakat: Transparansi informasi nilai properti'
            ],
            'technologies': {
                'backend': ['Python', 'Flask', 'PyMySQL', 'Scikit-learn'],
                'frontend': ['HTML5', 'CSS3', 'JavaScript', 'Bootstrap'],
                'gis': ['Leaflet.js', 'OpenStreetMap', 'QGIS'],
                'visualization': ['Plotly', 'Chart.js'],
                'database': ['MySQL']
            }
        }
        
        return render_template('tentang/index.html', **tentang_data)
        
    except Exception as e:
        flash(f'Error loading tentang page: {str(e)}', 'danger')
        print(f"Tentang error: {str(e)}")
        return render_template('tentang/index.html', 
                             research_title='Prediksi Nilai Zona Tanah Idi Rayeuk',
                             year='2025')   

#-------Prediksi API---------#
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Prediksi nilai zona tanah menggunakan model Random Forest hasil training.
    Menerima JSON atau form data dengan nama kolom sesuai dataset
    (nama_kelurahan, jenis_penggunaan_lahan, luas_zona_m2, ...).
    Fitur numerik yang dikosongkan diisi median zona di kelurahan yang sama.
    """
    input_data = request.get_json(silent=True) or request.form.to_dict()
    try:
        result = model_service.predict_land_value(input_data)
        return jsonify(result)
    except model_service.PredictionError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'success': False, 'error': 'Model gagal memproses prediksi'}), 500

#-------Hasil Analisis Route---------#
@app.route('/analisis')
@login_required
def analisis_results():
    """Halaman hasil analisis Random Forest (hasil evaluasi asli dari notebook)"""
    visualizations = load_visualizations()
    try:
        analysis_data = model_service.load_analysis_results()
        land_data = load_land_value_data()
        if land_data is not None and len(land_data) > 0:
            analysis_data['total_zones'] = len(land_data)
            analysis_data['total_kelurahan'] = land_data['nama_kelurahan'].nunique()
            analysis_data['data_source'] = 'Database + hasil evaluasi model'
        else:
            reference = model_service.load_reference_data()
            analysis_data['total_zones'] = len(reference)
            analysis_data['total_kelurahan'] = reference['nama_kelurahan'].nunique()
            analysis_data['data_source'] = 'Hasil evaluasi model (dataset training)'
    except Exception as e:
        flash(f'Error loading analysis results: {str(e)}', 'danger')
        print(f"Analysis error: {str(e)}")
        analysis_data = {
            'model_performance': None,
            'feature_importance': [],
            'category_importance': {},
            'prediction_results': [],
            'test_zone_ids': [],
            'yearly_error': [],
            'error_analysis': [],
            'model_config': {},
            'total_zones': 0,
            'total_kelurahan': 0,
            'data_source': 'Error - hasil model tidak dapat dimuat',
        }

    analysis_data.update({
        'visualizations': visualizations,
        'viz_count': len(visualizations),
        'features_count': 20,
        'algorithm': 'Random Forest Regression',
        'research_location': 'Kecamatan Idi Rayeuk, Aceh Timur',
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    })
    return render_template('analisis/index.html', **analysis_data)

# Route untuk serve visualization files
@app.route('/visualization/<filename>')
def serve_visualization(filename):
    """Serve visualization HTML files dari flask_components/visualizations"""
    if not filename.endswith('.html'):
        return "Visualization file not found", 404
    return send_from_directory(VISUALIZATION_DIR, filename)

#-------Error Handlers---------#
@app.errorhandler(404)
def not_found(error):
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'error': 'Endpoint tidak ditemukan'}), 404
    return render_template('layouts/404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'error': 'Terjadi kesalahan pada server'}), 500
    return render_template('layouts/500.html'), 500

#-------Context Processors---------#
@app.context_processor
def inject_global_vars():
    return {
        'app_name': 'Prediksi Nilai Zona Tanah',
        'app_description': 'Sistem Prediksi Nilai Zona Tanah di Kecamatan Idi Rayeuk, Aceh Timur menggunakan Random Forest Regression berbasis Sistem Informasi Geografis',
        'current_year': datetime.now().year,
        'is_logged_in': 'username' in session,
        'current_user': session.get('username', 'Guest'),
        'research_location': 'Kecamatan Idi Rayeuk, Aceh Timur',
        'research_algorithm': 'Random Forest Regression + GIS'
    }

if __name__ == '__main__':
    print("Starting Land Value Prediction System...")
    print("Research: Prediksi Nilai Zona Tanah Idi Rayeuk, Aceh Timur")
    print("Algorithm: Random Forest Regression + WebGIS")
    print("=" * 60)
    
    try:
        connection = get_db_connection()
        if connection:
            print("✅ Database connection: SUCCESS")
            cursor = connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM tbl_dataset")
            count = cursor.fetchone()[0]
            print(f"✅ Dataset records: {count}")
            cursor.close()
            connection.close()
        else:
            print("❌ Database connection: FAILED")
    except Exception as e:
        print(f"❌ Database error: {e}")

    try:
        metrics = model_service.get_model_metrics()
        model_service.load_pipeline()
        print(f"✅ Model Random Forest loaded (MAPE {metrics['mape']:.2f}%, R² {metrics['r2_score']:.4f})")
    except Exception as e:
        print(f"❌ Model error: {e}")
        print("   Pastikan scikit-learn sesuai requirements.txt dan file .pkl ada di flask_components/models/")

    print("=" * 60)
    print("Flask application starting...")
    print("URL: http://localhost:5000")
    print("Dataset URL: http://localhost:5000/dataset")
    print("Analysis URL: http://localhost:5000/analisis")
    print("=" * 60)
    
    app.run(
        debug=os.environ.get('FLASK_DEBUG', '0') == '1',
        host=os.environ.get('HOST', '127.0.0.1'),
        port=int(os.environ.get('PORT', 5000))
    )