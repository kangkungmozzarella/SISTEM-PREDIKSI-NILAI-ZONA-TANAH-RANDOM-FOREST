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

#-------Format angka Indonesia untuk template---------#
@app.template_filter('angka')
def format_angka(value, digits=0):
    """1234567.8 -> '1.234.568' (digits=0) atau '4,95' (digits=2)"""
    if value is None:
        return '-'
    text = f"{float(value):,.{digits}f}"
    return text.replace(',', '_').replace('.', ',').replace('_', '.')

@app.template_filter('rupiah')
def format_rupiah(value):
    return '-' if value is None else f"Rp {format_angka(value)}"

@app.template_filter('rupiah_singkat')
def format_rupiah_singkat(value):
    """68000 -> 'Rp 68 rb', 1250000 -> 'Rp 1,3 jt'"""
    if value is None:
        return '-'
    value = float(value)
    if value >= 1_000_000:
        return f"Rp {format_angka(value / 1_000_000, 1)} jt"
    return f"Rp {format_angka(value / 1000)} rb"

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

def get_land_data():
    """
    Data zona tanah dari database; jika database tidak tersedia atau kosong,
    memakai dataset training (flask_components/data). Mengembalikan (df, sumber).
    """
    land_data = load_land_value_data()
    if land_data is not None and len(land_data) > 0:
        return land_data, 'database'
    try:
        return model_service.load_reference_data().copy(), 'dataset_training'
    except Exception as e:
        print(f"Error loading reference data: {e}")
        return None, None

def summarize_land_data(df):
    """Ringkasan statistik nilai tanah yang dipakai di beberapa halaman"""
    values = df['nilai_tanah_per_m2'].astype(float)
    years = sorted(int(y) for y in df['tahun_data'].unique())
    land_types = []
    for jenis in ['Komersial', 'Pemukiman', 'Pertanian']:
        subset = df[df['jenis_penggunaan_lahan'] == jenis]
        if len(subset):
            land_types.append({
                'name': jenis,
                'count': int(len(subset)),
                'share': len(subset) / len(df),
                'avg': int(subset['nilai_tanah_per_m2'].astype(float).mean()),
            })
    return {
        'total_zones': int(len(df)),
        'total_kelurahan': int(df['nama_kelurahan'].nunique()),
        'total_locations': int(df[['koordinat_latitude', 'koordinat_longitude']].drop_duplicates().shape[0]),
        'years': years,
        'data_period': f"{years[0]}–{years[-1]}" if years else '-',
        'price_min': int(values.min()),
        'price_max': int(values.max()),
        'price_avg': int(values.mean()),
        'price_median': int(values.median()),
        'land_types': land_types,
    }

def get_model_summary():
    """Metrik, feature importance, dan konfigurasi model untuk halaman publik & admin"""
    try:
        results = model_service.load_analysis_results()
        return {
            'metrics': results['model_performance'],
            'feature_importance': results['feature_importance'],
            'category_importance': results['category_importance'],
            'model_config': results['model_config'],
            'train_size': results['train_size'],
            'test_size': results['test_size'],
        }
    except Exception as e:
        print(f"Error loading model summary: {e}")
        return None

@app.route('/')
def homepage():
    """Homepage publik: peta 3D, form prediksi, dan ringkasan model"""
    land_data, _ = get_land_data()
    try:
        kelurahan_list = model_service.get_kelurahan_list()
    except Exception as e:
        print(f"Error loading model: {e}")
        kelurahan_list = []

    return render_template(
        'index.html',
        summary=summarize_land_data(land_data) if land_data is not None else None,
        model=get_model_summary(),
        kelurahan_list=kelurahan_list,
    )

#-------Login---------#
@app.route('/admin/login', methods=['GET', 'POST'])
def login():
    """Login admin untuk dashboard"""
    if 'username' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')

        if (hmac.compare_digest(username, ADMIN_USERNAME)
                and hmac.compare_digest(password, ADMIN_PASSWORD)):
            session['username'] = username
            session['status'] = "Login"
            flash(f"Selamat datang, {username}.", 'success')
            return redirect(url_for('dashboard'))
        else:
            flash("Username atau password salah.", 'danger')

    return render_template('admin/login.html')

@app.route('/logout')
def logout():
    """Logout admin"""
    session.clear()
    flash('Anda sudah keluar.', 'info')
    return redirect(url_for('homepage'))

#-------Dashboard---------#
@app.route('/dashboard')
@login_required
def dashboard():
    """Ringkasan data dan model untuk admin"""
    land_data, source = get_land_data()
    if land_data is None:
        flash('Data zona tanah tidak dapat dimuat.', 'danger')
    return render_template(
        'dashboard/index.html',
        summary=summarize_land_data(land_data) if land_data is not None else None,
        model=get_model_summary(),
        data_source=source,
    )

#-------Dataset Management Routes---------#
@app.route('/dataset')
@login_required
def dataset_index():
    """Halaman untuk mengelola dataset zona tanah di database"""
    connection = get_db_connection()
    db_connected = connection is not None
    if connection:
        connection.close()

    land_data = load_land_value_data() if db_connected else None
    has_data = land_data is not None and len(land_data) > 0
    return render_template(
        'dataset/index.html',
        db_connected=db_connected,
        summary=summarize_land_data(land_data) if has_data else None,
        dataset=land_data.to_dict('records') if has_data else [],
    )

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
            skipped_rows = []
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
                    skipped_rows.append(csv_input.line_num)
                    continue
            
            connection.commit()
            cursor.close()
            connection.close()
            
            if inserted_count:
                flash(f'{inserted_count} baris berhasil diunggah.', 'success')
            if skipped_rows:
                preview = ', '.join(str(n) for n in skipped_rows[:10])
                more = ' dan lainnya' if len(skipped_rows) > 10 else ''
                flash(f'{len(skipped_rows)} baris dilewati karena kolom kosong atau formatnya salah (baris {preview}{more}).', 'warning')
        else:
            flash('File harus berformat .csv.', 'danger')
            
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
            flash('Semua data di tbl_dataset sudah dihapus.', 'success')
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

@app.route('/api/map/locations')
def api_map_locations():
    """
    Data untuk peta 3D: satu entri per titik koordinat, berisi ringkasan nilai tanah
    per jenis lahan per tahun. Memakai database; jika tidak tersedia, memakai
    dataset training di flask_components/data.
    """
    try:
        land_data = load_land_value_data()
        source = 'database'
        if land_data is None or len(land_data) == 0:
            land_data = model_service.load_reference_data().copy()
            source = 'dataset_training'

        land_data['koordinat_latitude'] = land_data['koordinat_latitude'].astype(float)
        land_data['koordinat_longitude'] = land_data['koordinat_longitude'].astype(float)
        land_data['tahun_data'] = land_data['tahun_data'].astype(int)
        land_data['nilai_tanah_per_m2'] = land_data['nilai_tanah_per_m2'].astype(float)

        locations = []
        group_cols = ['koordinat_latitude', 'koordinat_longitude', 'nama_kelurahan']
        for (lat, lng, kelurahan), group in land_data.groupby(group_cols):
            by_type = {}
            for jenis, type_group in group.groupby('jenis_penggunaan_lahan'):
                years = {}
                for tahun, year_group in type_group.groupby('tahun_data'):
                    values = year_group['nilai_tanah_per_m2']
                    years[int(tahun)] = {
                        'avg': round(float(values.mean())),
                        'min': int(values.min()),
                        'max': int(values.max()),
                        'count': int(len(values)),
                    }
                by_type[jenis] = {
                    'count': int(len(type_group)),
                    'avg': round(float(type_group['nilai_tanah_per_m2'].mean())),
                    'avg_luas_m2': round(float(type_group['luas_zona_m2'].astype(float).mean())),
                    'years': years,
                }
            locations.append({
                'lat': lat,
                'lng': lng,
                'nama_kelurahan': kelurahan,
                'total_zona': int(len(group)),
                'avg_aksesibilitas': round(float(group['aksesibilitas_skor'].astype(float).mean()), 1),
                'avg_jarak_pusat_km': round(float(group['jarak_pusat_kota_km'].astype(float).mean()), 2),
                'id_zona_list': sorted(group['id_zona'].astype(str).unique().tolist()),
                'by_type': by_type,
            })

        return jsonify({
            'status': 'success',
            'source': source,
            'total_locations': len(locations),
            'total_zona': int(len(land_data)),
            'years': sorted(land_data['tahun_data'].unique().tolist()),
            'value_range': {
                'min': int(land_data['nilai_tanah_per_m2'].min()),
                'max': int(land_data['nilai_tanah_per_m2'].max()),
            },
            'bounds': {
                'south': float(land_data['koordinat_latitude'].min()),
                'north': float(land_data['koordinat_latitude'].max()),
                'west': float(land_data['koordinat_longitude'].min()),
                'east': float(land_data['koordinat_longitude'].max()),
            },
            'data': locations,
        })
    except Exception as e:
        print(f"API map locations error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e), 'data': []}), 500

@app.route('/tentang')
def tentang():
    """Halaman tentang penelitian"""
    land_data, _ = get_land_data()
    return render_template(
        'tentang/index.html',
        summary=summarize_land_data(land_data) if land_data is not None else None,
        model=get_model_summary(),
        research_title='Prediksi Nilai Zona Tanah di Kecamatan Idi Rayeuk, Aceh Timur Menggunakan Metode Random Forest Regression Berbasis Sistem Informasi Geografis',
        research_location='Kecamatan Idi Rayeuk, Kabupaten Aceh Timur, Provinsi Aceh',
        main_objective='Mengembangkan sistem prediksi nilai zona tanah yang akurat menggunakan machine learning dan GIS untuk mendukung perencanaan tata ruang dan penilaian properti.',
        specific_objectives=[
            'Menganalisis faktor-faktor yang mempengaruhi nilai zona tanah',
            'Mengimplementasikan algoritma Random Forest untuk prediksi nilai tanah',
            'Mengintegrasikan sistem prediksi dengan platform WebGIS',
            'Mengevaluasi akurasi model prediksi yang dikembangkan',
            'Memberikan rekomendasi untuk perencanaan tata ruang wilayah',
        ],
        benefits=[
            ('Pemerintah daerah', 'Perencanaan tata ruang yang lebih baik'),
            ('Penilai properti', 'Estimasi nilai tanah yang objektif'),
            ('Investor', 'Analisis investasi properti berbasis data'),
            ('Akademisi', 'Referensi untuk penelitian selanjutnya'),
            ('Masyarakat', 'Transparansi informasi nilai tanah'),
        ],
        technologies=[
            ('Model', ['Python', 'scikit-learn', 'pandas', 'NumPy', 'Plotly']),
            ('Aplikasi web', ['Flask', 'PyMySQL', 'MySQL']),
            ('Peta', ['MapLibre GL JS', 'OpenStreetMap', 'Esri World Imagery']),
        ],
    )

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
            'test_errors': {},
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
