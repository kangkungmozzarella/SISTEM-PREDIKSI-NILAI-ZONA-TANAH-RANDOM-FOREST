import json

# Sesuaikan path file gadm41_IDN_4.json kamu
with open('C:\Users\HYPE\Documents\Orderan Joki\Belum Selesai\6 November 2025 (Iqbal)\random-forest\gadm41_IDN_4.json') as f:
    data = json.load(f)

# Filter hanya Idi Rayeuk
idi = [f for f in data['features'] 
       if f['properties'].get('NAME_3') == 'IdiRayeuk']

print(f'Jumlah gampong/kelurahan: {len(idi)}')

# Simpan
geojson = {'type': 'FeatureCollection', 'features': idi}
with open('C:/Users/NAMA_USER/Downloads/idi_rayeuk_desa.geojson', 'w') as f:
    json.dump(geojson, f)

print('Selesai! File tersimpan.')