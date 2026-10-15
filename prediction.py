"""
Layanan model Random Forest untuk prediksi nilai zona tanah.

Memuat pipeline hasil training (latih.ipynb, Cell 13) dari
flask_components/models/complete_pipeline.pkl dan hasil evaluasi dari
flask_components/results/. Preprocessing di sini harus sama persis dengan
Cell 5 di notebook (label encoding + kategori jarak & aksesibilitas).
"""
import json
import os
import pickle
from functools import lru_cache

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMPONENTS_DIR = os.path.join(BASE_DIR, 'flask_components')
PIPELINE_PATH = os.path.join(COMPONENTS_DIR, 'models', 'complete_pipeline.pkl')
REFERENCE_DATA_PATH = os.path.join(COMPONENTS_DIR, 'data', 'sorted_dataset.csv')
RESULTS_DIR = os.path.join(COMPONENTS_DIR, 'results')

NUMERIC_FEATURES = [
    'koordinat_latitude', 'koordinat_longitude', 'nomor_zona', 'luas_zona_m2',
    'jarak_pusat_kota_km', 'elevasi_mdpl', 'kemiringan_lereng_persen',
    'kepadatan_penduduk_km2', 'jarak_jalan_utama_m', 'jarak_sekolah_m',
    'jarak_puskesmas_m', 'jarak_pasar_m', 'status_listrik', 'status_air_bersih',
    'aksesibilitas_skor', 'tahun_data',
]
CATEGORICAL_FEATURES = ['nama_kelurahan', 'jenis_penggunaan_lahan']

FEATURE_LABELS = {
    'koordinat_latitude': 'Latitude',
    'koordinat_longitude': 'Longitude',
    'nomor_zona': 'Nomor zona',
    'luas_zona_m2': 'Luas zona',
    'jarak_pusat_kota_km': 'Jarak pusat kota',
    'elevasi_mdpl': 'Elevasi',
    'kemiringan_lereng_persen': 'Kemiringan lereng',
    'kepadatan_penduduk_km2': 'Kepadatan penduduk',
    'jarak_jalan_utama_m': 'Jarak jalan utama',
    'jarak_sekolah_m': 'Jarak sekolah',
    'jarak_puskesmas_m': 'Jarak puskesmas',
    'jarak_pasar_m': 'Jarak pasar',
    'status_listrik': 'Status listrik',
    'status_air_bersih': 'Status air bersih',
    'aksesibilitas_skor': 'Skor aksesibilitas',
    'tahun_data': 'Tahun data',
    'nama_kelurahan_encoded': 'Kelurahan',
    'jenis_penggunaan_lahan_encoded': 'Jenis lahan',
    'jarak_pusat_kategori_encoded': 'Kategori jarak pusat',
    'aksesibilitas_kategori_encoded': 'Kategori aksesibilitas',
}


class PredictionError(ValueError):
    """Input prediksi tidak valid."""


@lru_cache(maxsize=1)
def load_pipeline():
    with open(PIPELINE_PATH, 'rb') as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def load_reference_data():
    """Dataset training, dipakai untuk mengisi fitur yang tidak diinput user."""
    return pd.read_csv(REFERENCE_DATA_PATH)


def load_json_result(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_kelurahan_list():
    return list(load_pipeline()['label_encoders']['nama_kelurahan'].classes_)


def get_land_type_list():
    return list(load_pipeline()['label_encoders']['jenis_penggunaan_lahan'].classes_)


def get_model_metrics():
    """Metrik testing set (R², MAE, RMSE, MAPE) hasil notebook."""
    summary = load_json_result('final_performance_summary.json')
    if summary:
        return summary['final_metrics']
    return {k: float(v) for k, v in load_pipeline()['model_metrics'].items()}


def _default_values(kelurahan, jenis_lahan):
    """Median fitur numerik dari zona di kelurahan (dan jenis lahan) yang sama."""
    df = load_reference_data()
    subset = df[(df['nama_kelurahan'] == kelurahan) & (df['jenis_penggunaan_lahan'] == jenis_lahan)]
    if subset.empty:
        subset = df[df['nama_kelurahan'] == kelurahan]
    if subset.empty:
        subset = df
    return subset[NUMERIC_FEATURES].median().to_dict()


def _encode(encoder, value):
    if pd.isna(value):
        # Saat training, nilai di luar bin pd.cut menjadi NaN dan di-encode sebagai kelas tersendiri
        nan_idx = [i for i, c in enumerate(encoder.classes_) if pd.isna(c)]
        if nan_idx:
            return nan_idx[0]
        raise PredictionError('Nilai di luar rentang data training')
    if value not in encoder.classes_:
        raise PredictionError(f"Nilai '{value}' tidak dikenal oleh model")
    return int(encoder.transform([value])[0])


def _value_category(value):
    """Kategori nilai berdasarkan tertil nilai tanah di dataset training."""
    prices = load_reference_data()['nilai_tanah_per_m2']
    low, high = prices.quantile([1 / 3, 2 / 3])
    if value < low:
        return 'Nilai Rendah', 'low-value'
    if value < high:
        return 'Nilai Sedang', 'medium-value'
    return 'Nilai Tinggi', 'high-value'


def predict_land_value(input_data):
    """
    Prediksi nilai tanah per m² dari input dict.

    Wajib: nama_kelurahan, jenis_penggunaan_lahan.
    Fitur numerik yang kosong diisi median zona di kelurahan yang sama.
    """
    pipeline = load_pipeline()
    encoders = pipeline['label_encoders']

    kelurahan = (input_data.get('nama_kelurahan') or '').strip()
    jenis_lahan = (input_data.get('jenis_penggunaan_lahan') or '').strip()
    if not kelurahan or not jenis_lahan:
        raise PredictionError('Kelurahan dan jenis penggunaan lahan wajib diisi')

    row = _default_values(kelurahan, jenis_lahan)
    filled_defaults = []
    for feature in NUMERIC_FEATURES:
        raw = input_data.get(feature)
        if raw is None or raw == '':
            filled_defaults.append(feature)
            continue
        try:
            row[feature] = float(raw)
        except (TypeError, ValueError):
            raise PredictionError(f"Nilai {FEATURE_LABELS[feature]} harus berupa angka")
        if row[feature] < 0:
            raise PredictionError(f"Nilai {FEATURE_LABELS[feature]} tidak boleh negatif")

    if not 0 <= row['aksesibilitas_skor'] <= 10:
        raise PredictionError('Skor aksesibilitas harus antara 0 dan 10')

    row['nama_kelurahan_encoded'] = _encode(encoders['nama_kelurahan'], kelurahan)
    row['jenis_penggunaan_lahan_encoded'] = _encode(encoders['jenis_penggunaan_lahan'], jenis_lahan)

    # Sama dengan pd.cut di notebook Cell 5 (interval kanan-inklusif, batas bawah eksklusif).
    # Jarak 0 km tidak ada di data training dan tidak punya kelas NaN, jadi dianggap 'Sangat Dekat'.
    # Aksesibilitas 0 tetap menjadi NaN, sama seperti saat training.
    jarak_kategori = pd.cut([max(row['jarak_pusat_kota_km'], 1e-6)],
                            bins=[0, 1, 3, 6, float('inf')],
                            labels=['Sangat Dekat', 'Dekat', 'Sedang', 'Jauh'])[0]
    akses_kategori = pd.cut([row['aksesibilitas_skor']],
                            bins=[0, 3, 6, 8, 10],
                            labels=['Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi'])[0]
    row['jarak_pusat_kategori_encoded'] = _encode(encoders['jarak_pusat_kategori'], jarak_kategori)
    row['aksesibilitas_kategori_encoded'] = _encode(encoders['aksesibilitas_kategori'], akses_kategori)

    X = pd.DataFrame([row])[pipeline['feature_columns']]
    model = pipeline['model']
    prediction = float(model.predict(X)[0])

    # Sebaran prediksi antar pohon sebagai gambaran ketidakpastian
    tree_preds = np.array([tree.predict(X.values)[0] for tree in model.estimators_])
    p10, p90 = np.percentile(tree_preds, [10, 90])

    metrics = get_model_metrics()
    category, category_class = _value_category(prediction)

    return {
        'success': True,
        'prediction': round(prediction),
        'range': {
            'lower': round(max(0.0, prediction - metrics['mae'])),
            'upper': round(prediction + metrics['mae']),
        },
        'tree_range': {'p10': round(float(p10)), 'p90': round(float(p90))},
        'category': category,
        'category_class': category_class,
        'model_metrics': {k: float(v) for k, v in metrics.items()},
        'input_used': {k: (float(row[k]) if k in NUMERIC_FEATURES else row[k]) for k in NUMERIC_FEATURES},
        'filled_defaults': filled_defaults,
        'nama_kelurahan': kelurahan,
        'jenis_penggunaan_lahan': jenis_lahan,
    }


def load_analysis_results():
    """Kumpulkan hasil evaluasi asli dari flask_components/results untuk halaman analisis."""
    pipeline = load_pipeline()
    metrics = get_model_metrics()
    summary = load_json_result('final_performance_summary.json') or {}
    rf_results = load_json_result('random_forest_results.json') or {}
    error_analysis = load_json_result('error_analysis.json') or {}

    importances = pipeline['model'].feature_importances_
    feature_importance = sorted(
        [
            {
                'feature_name': name,
                'label': FEATURE_LABELS.get(name, name),
                'importance': float(imp),
            }
            for name, imp in zip(pipeline['feature_columns'], importances)
        ],
        key=lambda x: x['importance'],
        reverse=True,
    )
    category_importance = (load_json_result('feature_importance_analysis.json') or {}).get('category_importance', {})

    predictions_path = os.path.join(RESULTS_DIR, 'testing_predictions_detail.csv')
    prediction_results = []
    test_zone_ids = []
    if os.path.exists(predictions_path):
        test_df = pd.read_csv(predictions_path)
        test_zone_ids = test_df['id_zona'].tolist()
        test_df = test_df.sort_values('abs_error')
        for _, r in test_df.iterrows():
            prediction_results.append({
                'id_zona': r['id_zona'],
                'kelurahan': r['nama_kelurahan'],
                'tahun': int(r['tahun_data']),
                'jenis_lahan': r['jenis_penggunaan_lahan'],
                'actual_value': float(r['nilai_aktual']),
                'predicted_value': float(r['nilai_prediksi']),
                'absolute_error': float(r['abs_error']),
                'percentage_error': abs(float(r['pct_error'])),
            })

    model_info = summary.get('model_info', {})
    comparison = summary.get('comparison_train_test', {})

    return {
        'model_performance': {
            'mape': metrics['mape'],
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'r2_score': metrics['r2_score'],
        },
        'training_metrics': comparison.get('training_metrics'),
        'overfitting_check': comparison.get('overfitting_check'),
        'feature_importance': feature_importance,
        'category_importance': category_importance,
        'prediction_results': prediction_results,
        'test_zone_ids': test_zone_ids,
        'test_errors': {r['id_zona']: r['percentage_error'] for r in prediction_results},
        'yearly_error': error_analysis.get('yearly_error', []),
        'error_analysis': error_analysis.get('lahan_error', []),
        'model_config': rf_results.get('model_params', {}),
        'train_size': model_info.get('Training_Size'),
        'test_size': model_info.get('Testing_Size'),
    }
