"""
Ekstrak batas gampong/kelurahan Kecamatan Idi Rayeuk dari GeoJSON GADM level 4.

Pemakaian:
    python tes.py <path/gadm41_IDN_4.json> [output.geojson]

File GADM bisa diunduh dari https://gadm.org/download_country.html (Indonesia, GeoJSON).
"""
import json
import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'idi_rayeuk_desa.geojson'

    with open(input_path, encoding='utf-8') as f:
        data = json.load(f)

    # Filter hanya Idi Rayeuk
    idi = [f for f in data['features']
           if f['properties'].get('NAME_3') == 'IdiRayeuk']

    print(f'Jumlah gampong/kelurahan: {len(idi)}')

    geojson = {'type': 'FeatureCollection', 'features': idi}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(geojson, f)

    print(f'Selesai! File tersimpan di {output_path}')


if __name__ == '__main__':
    main()
