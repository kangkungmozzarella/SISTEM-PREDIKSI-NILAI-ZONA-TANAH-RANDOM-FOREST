/*
 * Peta 3D sebaran nilai zona tanah (MapLibre GL JS).
 *
 * Setiap titik koordinat di dataset digambar sebagai kolom 3D.
 *   - mode "value": tinggi & warna kolom = rata-rata nilai tanah/m² (bisa difilter tahun & jenis lahan)
 *   - mode "split": kolom bertumpuk jumlah data training (bawah) dan testing (atas)
 * Area peta dikunci ke sekitar titik-titik dataset.
 *
 * Pemakaian:
 *   ZoneMap3D.create({ container: 'landValueMap', mode: 'value', filterButtons: '.map-filter-btn', countEl: 'map-zone-count' });
 *   Opsi lain: showControls, showLegend, interactive, cooperativeGestures, pitch, bearing,
 *   testZoneIds & testErrors (mode split).
 *
 * Membutuhkan maplibre-gl (JS + CSS) sudah dimuat di halaman.
 */
(function (global) {
    'use strict';

    const API_URL = '/api/map/locations';
    const MAX_COLUMN_HEIGHT_M = 2600;
    const MAX_COLUMN_RADIUS_M = 210;
    const MIN_COLUMN_RADIUS_M = 35;
    const BOUNDS_PADDING_RATIO = 0.35;
    const LAND_TYPES = ['Komersial', 'Pemukiman', 'Pertanian'];
    // Warna diambil dari token CSS (static/css/app.css) agar konsisten & ikut mode gelap
    function cssVar(name, fallback) {
        const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
        return v || fallback;
    }
    function typeColor(type) {
        return cssVar('--land-' + type.toLowerCase(), { Komersial: '#c0452f', Pemukiman: '#2f6ea3', Pertanian: '#4f8a3c' }[type]);
    }
    function splitColors() {
        return { training: cssVar('--split-training', '#c0452f'), testing: cssVar('--split-testing', '#2f6ea3') };
    }
    const VALUE_COLORS = ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026'];

    const BASEMAPS = {
        peta: {
            label: 'Peta',
            tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
            attribution: '© <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors',
            maxzoom: 19,
        },
        satelit: {
            label: 'Satelit',
            tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'],
            attribution: 'Citra © Esri, Maxar, Earthstar Geographics',
            maxzoom: 18,
        },
    };

    const STYLE_ID = 'zone-map-3d-style';
    const CSS = `
        .zm3d-panel, .zm3d-legend {
            position: absolute; z-index: 2;
            background: var(--surface, #fff); color: var(--text, #1d2420);
            border: 1px solid var(--border, #e2dfd5); border-radius: 8px;
            box-shadow: var(--shadow-float, 0 6px 24px rgba(0,0,0,0.12));
            font-family: inherit;
        }
        .zm3d-panel {
            top: 10px; left: 10px; padding: 8px 10px; font-size: 0.78rem; max-width: calc(100% - 70px);
            display: flex; flex-direction: column; gap: 6px;
        }
        .zm3d-row { display: flex; align-items: center; gap: 4px; flex-wrap: wrap; }
        .zm3d-row-label { color: var(--text-muted, #5b6660); font-weight: 600; margin-right: 4px; min-width: 52px; }
        .zm3d-chip {
            font: inherit; font-size: 0.75rem; font-weight: 600; line-height: 1.5;
            border: 1px solid transparent; background: transparent; color: var(--text-muted, #5b6660);
            border-radius: 5px; padding: 2px 8px; cursor: pointer;
        }
        .zm3d-chip:hover { color: var(--text, #1d2420); background: var(--surface-2, #faf9f5); }
        .zm3d-chip.active { background: var(--primary-soft, #e4eee8); color: var(--primary, #24583f); }
        .zm3d-chip:focus-visible { outline: 2px solid var(--primary, #24583f); outline-offset: 1px; }
        .zm3d-legend { bottom: 30px; left: 10px; padding: 8px 12px; font-size: 0.75rem; min-width: 180px; }
        .zm3d-legend-title { font-weight: 700; margin-bottom: 4px; }
        .zm3d-gradient { height: 8px; border-radius: 4px; margin: 6px 0 3px; }
        .zm3d-gradient-labels { display: flex; justify-content: space-between; color: var(--text-muted, #5b6660); font-variant-numeric: tabular-nums; }
        .zm3d-legend-item { display: flex; align-items: center; gap: 6px; margin: 3px 0; }
        .zm3d-swatch { width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }
        .zm3d-hint { color: var(--text-subtle, #7d8780); margin-top: 4px; }
        .zm3d-status {
            position: absolute; inset: 0; z-index: 3; display: flex; align-items: center; justify-content: center;
            background: color-mix(in srgb, var(--surface, #fff) 80%, transparent);
            font-size: 0.9rem; color: var(--text-muted, #5b6660); text-align: center; padding: 16px;
        }
        .zm3d-popup .maplibregl-popup-content {
            padding: 0; border-radius: 8px; overflow: hidden; min-width: 270px;
            background: var(--surface, #fff); color: var(--text, #1d2420); font-family: inherit;
            box-shadow: var(--shadow-float, 0 6px 24px rgba(0,0,0,0.12));
        }
        .zm3d-popup.maplibregl-popup-anchor-bottom .maplibregl-popup-tip { border-top-color: var(--surface, #fff); }
        .zm3d-popup.maplibregl-popup-anchor-top .maplibregl-popup-tip { border-bottom-color: var(--surface, #fff); }
        .zm3d-popup.maplibregl-popup-anchor-left .maplibregl-popup-tip { border-right-color: var(--surface, #fff); }
        .zm3d-popup.maplibregl-popup-anchor-right .maplibregl-popup-tip { border-left-color: var(--surface, #fff); }
        .zm3d-popup-head { padding: 10px 32px 8px 12px; font-weight: 700; font-size: 0.9rem; border-bottom: 1px solid var(--border, #e2dfd5); }
        .zm3d-popup-head small { display: block; font-weight: 500; color: var(--text-muted, #5b6660); font-size: 0.75rem; }
        .zm3d-popup-body { padding: 8px 12px 10px; font-size: 0.78rem; }
        .zm3d-popup-body table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
        .zm3d-popup-body th { color: var(--text-muted, #5b6660); font-weight: 600; text-align: left; padding: 3px 0; }
        .zm3d-popup-body td { padding: 3px 0; border-top: 1px solid var(--border, #e2dfd5); }
        .zm3d-popup-body td.num, .zm3d-popup-body th.num { text-align: right; padding-left: 10px; white-space: nowrap; }
        .zm3d-popup-row { display: flex; justify-content: space-between; gap: 12px; padding: 3px 0; }
        .zm3d-popup-row span:first-child { color: var(--text-muted, #5b6660); }
        .zm3d-popup-row span:last-child { font-weight: 700; white-space: nowrap; font-variant-numeric: tabular-nums; }
        .zm3d-popup-row.highlight span:last-child { color: var(--accent, #b0502a); }
        .zm3d-popup .maplibregl-popup-close-button { color: var(--text-muted, #5b6660); font-size: 18px; width: 28px; height: 28px; line-height: 28px; padding: 0; }
        .zm3d-popup .maplibregl-popup-close-button:hover { background: var(--surface-2, #faf9f5); color: var(--text, #1d2420); }
        .maplibregl-ctrl-group { background: var(--surface, #fff); border: 1px solid var(--border, #e2dfd5); box-shadow: none !important; }
        .maplibregl-ctrl-group button { background-color: var(--surface, #fff); }
        .maplibregl-ctrl-group button + button { border-top-color: var(--border, #e2dfd5); }
        .maplibregl-ctrl-group button .maplibregl-ctrl-icon {
            filter: var(--map-control-icon-filter, none);
        }
        :root[data-theme="dark"] .maplibregl-ctrl-group button .maplibregl-ctrl-icon {
            filter: invert(1) brightness(1.8);
        }
        @media (prefers-color-scheme: dark) {
            :root:not([data-theme="light"]) .maplibregl-ctrl-group button .maplibregl-ctrl-icon {
                filter: invert(1) brightness(1.8);
            }
        }
        :root[data-theme="dark"] .maplibregl-ctrl-group button {
            background-color: var(--surface, #181d1a);
        }
        @media (prefers-color-scheme: dark) {
            :root:not([data-theme="light"]) .maplibregl-ctrl-group button {
                background-color: var(--surface, #181d1a);
            }
        }
        @media (max-width: 576px) {
            .zm3d-legend { bottom: 28px; min-width: 0; width: 150px; padding: 6px 8px; font-size: 0.7rem; }
            .zm3d-legend .zm3d-hint { display: none; }
            .zm3d-panel { padding: 6px 8px; }
            .zm3d-row-label { min-width: 0; }
            .zm3d-chip { padding: 1px 6px; font-size: 0.7rem; }
            .maplibregl-ctrl-group button { width: 36px; height: 36px; }
            .maplibregl-ctrl-top-right { top: 6px; right: 6px; }
        }
    `;

    function injectStyles() {
        if (document.getElementById(STYLE_ID)) return;
        const style = document.createElement('style');
        style.id = STYLE_ID;
        style.textContent = CSS;
        document.head.appendChild(style);
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text == null ? '' : String(text);
        return div.innerHTML;
    }

    function formatRupiah(value) {
        return 'Rp ' + Math.round(value).toLocaleString('id-ID');
    }

    function formatShortRupiah(value) {
        if (value >= 1e6) return 'Rp ' + (value / 1e6).toLocaleString('id-ID', { maximumFractionDigits: 1 }) + ' jt';
        return 'Rp ' + Math.round(value / 1000).toLocaleString('id-ID') + ' rb';
    }

    function metersToDegrees(lat, meters) {
        return {
            lat: meters / 111320,
            lng: meters / (111320 * Math.cos(lat * Math.PI / 180)),
        };
    }

    function distanceMeters(a, b) {
        const dLat = (a.lat - b.lat) * 111320;
        const dLng = (a.lng - b.lng) * 111320 * Math.cos(((a.lat + b.lat) / 2) * Math.PI / 180);
        return Math.hypot(dLat, dLng);
    }

    function hexagon(lat, lng, radiusM) {
        const deg = metersToDegrees(lat, radiusM);
        const ring = [];
        for (let i = 0; i <= 6; i++) {
            const angle = (Math.PI / 3) * (i % 6) + Math.PI / 6;
            ring.push([lng + deg.lng * Math.cos(angle), lat + deg.lat * Math.sin(angle)]);
        }
        return { type: 'Polygon', coordinates: [ring] };
    }

    // Kolom tidak boleh bertabrakan dengan titik terdekat
    function assignRadii(locations) {
        locations.forEach(loc => {
            let nearest = Infinity;
            locations.forEach(other => {
                if (other !== loc) nearest = Math.min(nearest, distanceMeters(loc, other));
            });
            loc._radius = Math.max(MIN_COLUMN_RADIUS_M, Math.min(MAX_COLUMN_RADIUS_M, nearest * 0.42));
        });
    }

    // Rata-rata tertimbang (jumlah zona) untuk tahun & jenis lahan aktif
    function aggregateValue(location, year, landType) {
        let total = 0;
        let count = 0;
        Object.entries(location.by_type).forEach(([type, info]) => {
            if (landType !== 'Semua' && type !== landType) return;
            const entries = year === 'Semua' ? Object.values(info.years) : [info.years[year]].filter(Boolean);
            entries.forEach(e => {
                total += e.avg * e.count;
                count += e.count;
            });
        });
        return count ? { value: total / count, count } : null;
    }

    function create(options) {
        injectStyles();

        const opts = Object.assign({
            mode: 'value',
            showControls: true,
            showLegend: true,
            interactive: true,
            pitch: 55,
            bearing: -20,
            testZoneIds: [],
            testErrors: {},
        }, options);

        const containerEl = document.getElementById(opts.container);
        if (!containerEl) return null;
        if (!global.maplibregl) {
            containerEl.innerHTML = '<div class="zm3d-status">Library peta gagal dimuat. Periksa koneksi internet.</div>';
            return null;
        }
        containerEl.style.position = containerEl.style.position || 'relative';

        const state = {
            year: null,
            landType: 'Semua',
            basemap: 'peta',
            data: null,
            popup: null,
        };
        const testIds = new Set(opts.testZoneIds || []);

        const statusEl = document.createElement('div');
        statusEl.className = 'zm3d-status';
        statusEl.textContent = 'Memuat peta 3D…';

        const map = new maplibregl.Map({
            container: containerEl,
            style: {
                version: 8,
                sources: Object.fromEntries(Object.entries(BASEMAPS).map(([key, b]) => [
                    'basemap-' + key,
                    { type: 'raster', tiles: b.tiles, tileSize: 256, attribution: b.attribution, maxzoom: b.maxzoom },
                ])),
                layers: Object.keys(BASEMAPS).map(key => ({
                    id: 'basemap-' + key,
                    type: 'raster',
                    source: 'basemap-' + key,
                    layout: { visibility: key === 'peta' ? 'visible' : 'none' },
                })),
            },
            center: [97.765, 4.945],
            zoom: 11,
            pitch: opts.pitch,
            bearing: opts.bearing,
            maxPitch: 75,
            interactive: opts.interactive,
            attributionControl: { compact: true },
            cooperativeGestures: !!opts.cooperativeGestures,
        });
        containerEl.appendChild(statusEl);

        // Container bisa berubah ukuran setelah layout/font selesai dimuat
        if (global.ResizeObserver) {
            new ResizeObserver(() => map.resize()).observe(containerEl);
        }

        if (opts.interactive) {
            map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'top-right');
            map.addControl(new maplibregl.FullscreenControl(), 'top-right');
        }

        function setStatus(text) {
            if (text) {
                statusEl.textContent = text;
                statusEl.style.display = 'flex';
            } else {
                statusEl.style.display = 'none';
            }
        }

        function buildValueFeatures() {
            const features = [];
            let shownLocations = 0;
            let shownZones = 0;
            state.data.data.forEach((loc, idx) => {
                const agg = aggregateValue(loc, state.year, state.landType);
                if (!agg) return;
                shownLocations += 1;
                shownZones += agg.count;
                features.push({
                    type: 'Feature',
                    id: idx,
                    geometry: hexagon(loc.lat, loc.lng, loc._radius),
                    properties: {
                        idx,
                        value: agg.value,
                        height: (agg.value / state.data.value_range.max) * MAX_COLUMN_HEIGHT_M,
                        base: 0,
                        color: '',
                    },
                });
            });
            return { features, shownLocations, shownZones };
        }

        function buildSplitFeatures() {
            const features = [];
            let maxTotal = 1;
            state.data.data.forEach(loc => { maxTotal = Math.max(maxTotal, loc.total_zona); });
            // Lebih pendek dari mode nilai: hampir semua titik punya jumlah zona serupa
            const unit = (MAX_COLUMN_HEIGHT_M * 0.6) / maxTotal;
            state.data.data.forEach((loc, idx) => {
                const testCount = loc.id_zona_list.filter(id => testIds.has(id)).length;
                const trainCount = loc.total_zona - testCount;
                const geometry = hexagon(loc.lat, loc.lng, loc._radius);
                if (trainCount > 0) {
                    features.push({ type: 'Feature', geometry, properties: { idx, base: 0, height: trainCount * unit, color: splitColors().training } });
                }
                if (testCount > 0) {
                    features.push({ type: 'Feature', geometry, properties: { idx, base: trainCount * unit, height: loc.total_zona * unit, color: splitColors().testing } });
                }
            });
            return { features, shownLocations: state.data.data.length, shownZones: state.data.total_zona };
        }

        function colorExpression() {
            if (opts.mode === 'split') return ['get', 'color'];
            const { min, max } = state.data.value_range;
            const stops = [];
            VALUE_COLORS.forEach((color, i) => {
                stops.push(min + ((max - min) * i) / (VALUE_COLORS.length - 1), color);
            });
            return ['interpolate', ['linear'], ['get', 'value'], ...stops];
        }

        function render() {
            const result = opts.mode === 'split' ? buildSplitFeatures() : buildValueFeatures();
            const geojson = { type: 'FeatureCollection', features: result.features };
            const source = map.getSource('zones');
            if (source) {
                source.setData(geojson);
            } else {
                map.addSource('zones', { type: 'geojson', data: geojson });
                map.addLayer({
                    id: 'zones-3d',
                    type: 'fill-extrusion',
                    source: 'zones',
                    paint: {
                        'fill-extrusion-color': colorExpression(),
                        'fill-extrusion-height': ['get', 'height'],
                        'fill-extrusion-base': ['get', 'base'],
                        'fill-extrusion-opacity': 0.9,
                        'fill-extrusion-vertical-gradient': true,
                    },
                });
            }
            if (opts.countEl) {
                const el = document.getElementById(opts.countEl);
                if (el) el.textContent = `${result.shownLocations} titik · ${result.shownZones} zona ditampilkan`;
            }
            if (state.popup) {
                state.popup.remove();
                state.popup = null;
            }
        }

        function valuePopupHtml(loc) {
            const years = state.data.years;
            const types = LAND_TYPES.filter(t => loc.by_type[t]);
            const agg = aggregateValue(loc, state.year, state.landType);
            const header = years.map(y => `<th class="num">${y}</th>`).join('');
            const rows = types.map(type => {
                const info = loc.by_type[type];
                const cells = years.map(y => {
                    const e = info.years[y];
                    return `<td class="num">${e ? formatShortRupiah(e.avg) : '–'}</td>`;
                }).join('');
                return `<tr><td><span class="zm3d-swatch" style="display:inline-block;vertical-align:middle;margin-right:4px;background:${typeColor(type)}"></span>${type} <span style="color:#999">(${info.count})</span></td>${cells}</tr>`;
            }).join('');
            const filterLabel = [
                state.landType === 'Semua' ? 'semua jenis lahan' : state.landType,
                state.year === 'Semua' ? 'semua tahun' : 'tahun ' + state.year,
            ].join(', ');
            return `
                <div class="zm3d-popup-head">${escapeHtml(loc.nama_kelurahan)}
                    <small>${loc.total_zona} zona · aksesibilitas ${loc.avg_aksesibilitas}/10 · ${loc.avg_jarak_pusat_km} km ke pusat</small>
                </div>
                <div class="zm3d-popup-body">
                    ${agg ? `<div class="zm3d-popup-row highlight"><span>Rata-rata (${escapeHtml(filterLabel)})</span><span>${formatRupiah(agg.value)}/m²</span></div>` : ''}
                    <table style="margin-top:6px;">
                        <thead><tr><th>Rata-rata/m²</th>${header}</tr></thead>
                        <tbody>${rows}</tbody>
                    </table>
                </div>`;
        }

        function splitPopupHtml(loc) {
            const testRows = loc.id_zona_list.filter(id => testIds.has(id));
            const errors = testRows.map(id => opts.testErrors[id]).filter(v => v != null);
            const mape = errors.length ? errors.reduce((a, b) => a + b, 0) / errors.length : null;
            return `
                <div class="zm3d-popup-head">${escapeHtml(loc.nama_kelurahan)}<small>${loc.total_zona} zona</small></div>
                <div class="zm3d-popup-body">
                    <div class="zm3d-popup-row"><span>Data training</span><span>${loc.total_zona - testRows.length}</span></div>
                    <div class="zm3d-popup-row"><span>Data testing</span><span>${testRows.length}</span></div>
                    ${mape != null ? `<div class="zm3d-popup-row"><span>MAPE testing di titik ini</span><span>${mape.toFixed(2)}%</span></div>` : ''}
                </div>`;
        }

        function buildLegend() {
            const legend = document.createElement('div');
            legend.className = 'zm3d-legend';
            if (opts.mode === 'split') {
                legend.innerHTML = `
                    <div class="zm3d-legend-title">Data Analisis</div>
                    <div class="zm3d-legend-item"><span class="zm3d-swatch" style="background:${splitColors().training}"></span>Training</div>
                    <div class="zm3d-legend-item"><span class="zm3d-swatch" style="background:${splitColors().testing}"></span>Testing</div>
                    <div class="zm3d-hint">Tinggi kolom = jumlah zona</div>`;
            } else {
                const { min, max } = state.data.value_range;
                legend.innerHTML = `
                    <div class="zm3d-legend-title">Rata-rata nilai tanah/m²</div>
                    <div class="zm3d-gradient" style="background:linear-gradient(90deg, ${VALUE_COLORS.join(', ')})"></div>
                    <div class="zm3d-gradient-labels"><span>${formatShortRupiah(min)}</span><span>${formatShortRupiah(max)}</span></div>
                    <div class="zm3d-hint">Tinggi & warna kolom = nilai tanah</div>`;
            }
            if (opts.interactive) {
                legend.innerHTML += '<div class="zm3d-hint">Klik kolom untuk detail · Ctrl/klik kanan + geser untuk memutar</div>';
            }
            containerEl.appendChild(legend);
        }

        function buildPanel() {
            const panel = document.createElement('div');
            panel.className = 'zm3d-panel';

            if (opts.mode === 'value') {
                const yearRow = document.createElement('div');
                yearRow.className = 'zm3d-row';
                yearRow.innerHTML = '<span class="zm3d-row-label">Tahun</span>';
                ['Semua', ...state.data.years].forEach(year => {
                    const btn = document.createElement('button');
                    btn.type = 'button';
                    btn.className = 'zm3d-chip' + (String(year) === String(state.year) ? ' active' : '');
                    btn.textContent = year;
                    btn.setAttribute('aria-pressed', String(year) === String(state.year));
                    btn.addEventListener('click', () => {
                        state.year = year;
                        yearRow.querySelectorAll('.zm3d-chip').forEach(b => {
                            b.classList.toggle('active', b === btn);
                            b.setAttribute('aria-pressed', b === btn);
                        });
                        render();
                    });
                    yearRow.appendChild(btn);
                });
                panel.appendChild(yearRow);
            }

            const viewRow = document.createElement('div');
            viewRow.className = 'zm3d-row';
            viewRow.innerHTML = '<span class="zm3d-row-label">Tampilan</span>';
            Object.entries(BASEMAPS).forEach(([key, b]) => {
                const btn = document.createElement('button');
                btn.type = 'button';
                btn.className = 'zm3d-chip' + (key === state.basemap ? ' active' : '');
                btn.textContent = b.label;
                btn.setAttribute('aria-pressed', key === state.basemap);
                btn.addEventListener('click', () => {
                    state.basemap = key;
                    Object.keys(BASEMAPS).forEach(k => {
                        map.setLayoutProperty('basemap-' + k, 'visibility', k === key ? 'visible' : 'none');
                    });
                    viewRow.querySelectorAll('.zm3d-chip[data-basemap]').forEach(b2 => {
                        b2.classList.toggle('active', b2 === btn);
                        b2.setAttribute('aria-pressed', b2 === btn);
                    });
                });
                btn.dataset.basemap = key;
                viewRow.appendChild(btn);
            });
            const resetBtn = document.createElement('button');
            resetBtn.type = 'button';
            resetBtn.className = 'zm3d-chip';
            resetBtn.textContent = 'Reset';
            resetBtn.title = 'Kembalikan sudut pandang awal';
            resetBtn.addEventListener('click', () => fitToData(true));
            viewRow.appendChild(resetBtn);
            panel.appendChild(viewRow);

            containerEl.appendChild(panel);
        }

        function bindFilterButtons() {
            if (!opts.filterButtons) return;
            const buttons = document.querySelectorAll(opts.filterButtons);
            buttons.forEach(btn => {
                btn.addEventListener('click', function () {
                    buttons.forEach(b => b.classList.remove('active'));
                    this.classList.add('active');
                    state.landType = this.dataset.type || 'Semua';
                    render();
                });
            });
        }

        function dataBounds() {
            const b = state.data.bounds;
            return [[b.west, b.south], [b.east, b.north]];
        }

        function fitToData(animate) {
            map.fitBounds(dataBounds(), {
                padding: { top: 60, bottom: 60, left: 40, right: 40 },
                pitch: opts.pitch,
                bearing: opts.bearing,
                animate: !!animate && !window.matchMedia('(prefers-reduced-motion: reduce)').matches,
            });
        }

        function lockToDataArea() {
            const b = state.data.bounds;
            const padLat = Math.max((b.north - b.south) * BOUNDS_PADDING_RATIO, 0.02);
            const padLng = Math.max((b.east - b.west) * BOUNDS_PADDING_RATIO, 0.02);
            map.setMaxBounds([[b.west - padLng, b.south - padLat], [b.east + padLng, b.north + padLat]]);
            map.setMinZoom(Math.max(map.getZoom() - 1, 9));
        }

        // Data diambil paralel dengan pemuatan peta; kolom digambar begitu style siap,
        // tanpa menunggu semua tile basemap selesai (event 'load').
        const styleReady = new Promise(resolve => {
            if (map.style && map.style._loaded) resolve();
            else map.once('style.load', resolve);
        });
        const dataReady = fetch(API_URL).then(res => res.json());

        Promise.all([dataReady, styleReady])
                .then(([response]) => {
                    if (response.status !== 'success' || !response.data.length) {
                        setStatus('Tidak ada data zona untuk ditampilkan.');
                        return;
                    }
                    state.data = response;
                    state.year = opts.mode === 'value' ? response.years[response.years.length - 1] : 'Semua';
                    assignRadii(response.data);

                    fitToData(false);
                    lockToDataArea();
                    render();
                    if (opts.showLegend) buildLegend();
                    if (opts.showControls) buildPanel();
                    bindFilterButtons();
                    setStatus(null);

                    if (opts.interactive) {
                        map.on('click', 'zones-3d', e => {
                            const feature = e.features && e.features[0];
                            if (!feature) return;
                            const loc = state.data.data[feature.properties.idx];
                            if (state.popup) state.popup.remove();
                            state.popup = new maplibregl.Popup({ className: 'zm3d-popup', maxWidth: '340px' })
                                .setLngLat([loc.lng, loc.lat])
                                .setHTML(opts.mode === 'split' ? splitPopupHtml(loc) : valuePopupHtml(loc))
                                .addTo(map);
                        });
                        map.on('mouseenter', 'zones-3d', () => { map.getCanvas().style.cursor = 'pointer'; });
                        map.on('mouseleave', 'zones-3d', () => { map.getCanvas().style.cursor = ''; });
                    }
                })
                .catch(err => {
                    console.error('Error loading 3D map:', err);
                    setStatus('Gagal memuat data peta.');
                });

        return { map, render };
    }

    global.ZoneMap3D = { create };
})(window);
