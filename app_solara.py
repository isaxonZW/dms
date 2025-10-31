# app_solara.py
import io, os, json, zipfile, tempfile, datetime, requests, math
import base64
import functools
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import solara as sl
import solara.lab as lab
import reacton.ipywidgets as w
from pyproj import Transformer
import pandas as pd
import geopandas as gpd
from shapely import wkb
from fpdf import FPDF

import ee
import geemap
from ipyleaflet import Map as LeafletMap, TileLayer, LayersControl, basemaps, basemap_to_tiles, ImageOverlay

# -----------------------
# EE INIT (one-time interactive done externally)
# -----------------------
def ee_init_once():
    try:
        geemap.ee_initialize()
    except Exception:
        ee.Initialize()
ee_init_once()

# -----------------------
# CONSTANTS
# -----------------------
COUNTRY_NAME = "South Africa"
HECTARE_THRESHOLD = 2500
KM2_THRESHOLD = HECTARE_THRESHOLD / 100.0   # 25 km2

# Hardcoded Dates (Removed from UI)
DEFAULT_BASELINE = ("2003-01-01", "2020-12-31")  # CHIRPS coverage
DEFAULT_ANALYSIS = ("2022-01-01", datetime.date.today().strftime("%Y-%m-%d"))

# Hardcoded Scales (Removed from UI)
NDVI_SCALE_M = 2000
LST_SCALE_M  = 2000
SPI_SCALE_M  = 5000

# Palettes
PALETTE_NDVI    = ['#654321','#b08d57','#ffffbf','#a6d96a','#1a9641']  # 0..1
PALETTE_SPI     = ['#67001f','#b2182b','#d6604d','#f4a582','#fddbc7',
                   '#d1e5f0','#92c5de','#4393c3','#2166ac','#053061']  # dry..wet
PALETTE_PERC    = ['#a50026','#f46d43','#fee08b','#abdda4','#1a9850']   # 0..100
PALETTE_VHI_CLASS = ['#8b0000','#ff7f00','#ffd700','#9acd32','#1a9850'] # 1..5
# SPI classes (approx): <=-2, -2..-1.5, -1.5..-1, -1..-0.5, -0.5..0.5, 0.5..1, 1..1.5, 1.5..2, >2

NDVI_COLL_ID = "MODIS/061/MOD13A3"             # monthly NDVI (scale 0.0001)
CHIRPS_MO_ID = "UCSB-CHG/CHIRPS/Monthly"       # monthly precip 'precipitation'
LST_COLL_ID  = "MODIS/061/MOD11A2"             # 8-day LST (K * 0.02)

# ASIS ImageServer endpoints (public)
ASIS_VCI_M = "https://asis-esri.fao.org/image/rest/services/VCI_M/ImageServer"
ASIS_VHI_M = "https://asis-esri.fao.org/image/rest/services/VHI_M/ImageServer"

# -----------------------
# UTIL: geometry cleaning & scale picking
#   (adapted from your Streamlit code)
# -----------------------
def make_valid_safe(geom):
    try:
        from shapely.validation import make_valid as _make_valid
        return _make_valid(geom)
    except Exception:
        try:
            return geom.buffer(0)
        except Exception:
            return geom

def simplify_roi_for_stats(roi: ee.Geometry, base_scale_m: int) -> ee.Geometry:
    err_margin = max(int(base_scale_m), 30)
    max_error  = max(int(base_scale_m * 2), 60)
    try:
        cleaned = roi.buffer(0, err_margin)
    except Exception:
        cleaned = roi
    try:
        return cleaned.simplify(max_error)
    except Exception:
        healed = cleaned.buffer(err_margin, err_margin).buffer(-err_margin, err_margin)
        return healed.simplify(max_error)

def pick_working_scale(roi: ee.Geometry, base_scale_m: int) -> int:
    area_km2 = ee.Number(roi.area(maxError=1)).divide(1e6).getInfo()
    if area_km2 > 150_000:
        return int(base_scale_m * 4)
    if area_km2 > 50_000:
        return int(base_scale_m * 2)
    return int(base_scale_m)

def pick_display_scale(area_km2: float) -> int:
    if area_km2 > 150_000: return 20000
    if area_km2 >  80_000: return 15000
    if area_km2 >  40_000: return 10000
    if area_km2 >  20_000: return  7000
    if area_km2 >   5_000: return  5000
    return 3000

# -----------------------
# ADMIN BOUNDARIES
# (FAO GAUL, same as your app)
# -----------------------
def gaul_level1():
    return ee.FeatureCollection("FAO/GAUL/2015/level1").filter(ee.Filter.eq("ADM0_NAME", COUNTRY_NAME))
def gaul_level2():
    return ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq("ADM0_NAME", COUNTRY_NAME))

@functools.lru_cache(maxsize=None)
def list_provinces() -> List[str]:
    return sorted(list(set(gaul_level1().aggregate_array("ADM1_NAME").getInfo())))

@functools.lru_cache(maxsize=None)
def list_districts(province: str) -> List[str]:
    ft = gaul_level2().filter(ee.Filter.eq("ADM1_NAME", province))
    return sorted(list(set(ft.aggregate_array("ADM2_NAME").getInfo())))

def get_roi_admin(level: str, province: Optional[str]=None, district: Optional[str]=None) -> Tuple[ee.Geometry, str]:
    if level == "country":
        return gaul_level1().geometry(), COUNTRY_NAME
    if level == "province" and province:
        return gaul_level1().filter(ee.Filter.eq("ADM1_NAME", province)).geometry(), f"Province: {province}"
    if level == "district" and province and district:
        g = gaul_level2().filter(ee.Filter.And(
                ee.Filter.eq("ADM1_NAME", province),
                ee.Filter.eq("ADM2_NAME", district)
            )).geometry()
        return g, f"District: {district}, {province}"
    raise ValueError("Invalid admin selection")

# -----------------------
# UPLOAD ROI (zip shp / geojson / kml)
# (ported and slimmed; keeps your robustness)
# -----------------------
def roi_from_uploaded(files) -> Tuple[ee.Geometry, str]:
    if not files:
        raise ValueError("No files")
    tmpdir = tempfile.TemporaryDirectory()
    tmp = Path(tmpdir.name)

    def _read_any_to_gdf(path: Path) -> gpd.GeoDataFrame:
        try:
            gdf = gpd.read_file(path, engine="pyogrio")
        except Exception:
            gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        else:
            gdf = gdf.to_crs(4326)
        # clean/simplify
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()].copy()
        gdf["geometry"] = gdf["geometry"].apply(make_valid_safe)
        if len(gdf) > 1:
            gdf["_diss"] = 1
            gdf = gdf.dissolve("_diss", as_index=False)
        gdf["geometry"] = gdf["geometry"].simplify(0.0005, preserve_topology=True)  # ~55m
        return gdf

    saved = []
    for f in files:
        p = tmp / f["name"]
        p.write_bytes(f["data"].read())
        saved.append(p)

    if len(saved) == 1 and saved[0].suffix.lower() == ".zip":
        with zipfile.ZipFile(saved[0]) as z:
            z.extractall(tmp)
        cands = list(tmp.rglob("*.geojson")) + list(tmp.rglob("*.json")) + list(tmp.rglob("*.kml"))
        vector_path = cands[0] if cands else (list(tmp.rglob("*.shp"))[0])
    else:
        cands = [p for p in saved if p.suffix.lower() in [".geojson",".json",".kml",".shp"]]
        vector_path = cands[0]

    gdf = _read_any_to_gdf(vector_path)
    if gdf.empty:
        raise ValueError("Uploaded file has no features.")
    try:
        fc = geemap.geopandas_to_ee(gdf.reset_index(drop=True), geodesic=False)
        return fc.geometry(), "Uploaded ROI"
    except Exception:
        try:
            gj = json.loads(gdf.to_json())
            geom = ee.Geometry(gj["features"][0]["geometry"], proj=None, geodesic=False)
            return geom, "Uploaded ROI"
        except Exception:
            minx, miny, maxx, maxy = gdf.total_bounds
            bbox = ee.Geometry.Rectangle([minx, miny, maxx, maxy], proj="EPSG:4326", geodesic=False)
            return bbox, "Uploaded ROI (BBox)"

# -----------------------
# DATASETS
# -----------------------
def monthly_ndvi_collection(start: str, end: str, roi: ee.Geometry) -> ee.ImageCollection:
    col = (ee.ImageCollection(NDVI_COLL_ID)
           .filterDate(start, end)
           .filterBounds(roi)
           .select("NDVI"))
    def _scale_tag(img):
        ndvi = img.multiply(0.0001).rename("NDVI")
        return (ndvi.set("system:time_start", img.get("system:time_start"))
                    .set("month", ee.Date(img.get("system:time_start")).get("month")))
    return col.map(_scale_tag)

def monthly_lstC_collection(start: str, end: str, roi: ee.Geometry) -> ee.ImageCollection:
    eight_day = (ee.ImageCollection(LST_COLL_ID)
                 .filterDate(start, end)
                 .filterBounds(roi)
                 .select("LST_Day_1km"))
    def _toC(img):
        lstC = img.multiply(0.02).subtract(273.15).rename("LST_C")
        return lstC.set("system:time_start", img.get("system:time_start"))
    eight_dayC = eight_day.map(_toC)

    s = ee.Date(start); e = ee.Date(end)
    nmonths = e.difference(s, "month").round()
    seq = ee.List.sequence(0, nmonths.subtract(1))
    def _monthly(i):
        start_m = s.advance(i, "month"); end_m = start_m.advance(1, "month")
        m = eight_dayC.filterDate(start_m, end_m).mean().rename("LST_C")
        return m.set("system:time_start", start_m.millis()).set("month", start_m.get("month"))
    return ee.ImageCollection(seq.map(_monthly))

def compute_vci(ndvi_monthly: ee.ImageCollection, baseline_start: str=DEFAULT_BASELINE[0], baseline_end: str=DEFAULT_BASELINE[1]) -> ee.ImageCollection:
    baseline = ndvi_monthly.filterDate(baseline_start, baseline_end)
    def _vci(img):
        month = ee.Number(img.get("month"))
        b = baseline.filter(ee.Filter.calendarRange(month, month, "month"))
        ndvi_min = b.min().rename("NDVI_min")
        ndvi_max = b.max().rename("NDVI_max")
        ndvi = img.select("NDVI")
        denom = ndvi_max.subtract(ndvi_min)
        vci = ndvi.subtract(ndvi_min).divide(denom).multiply(100).rename("VCI")
        vci = vci.where(denom.eq(0), 0)
        return vci.set("system:time_start", img.get("system:time_start")).set("month", month)
    return ndvi_monthly.map(_vci)

def compute_tci(lst_monthly: ee.ImageCollection, baseline_start: str=DEFAULT_BASELINE[0], baseline_end: str=DEFAULT_BASELINE[1]) -> ee.ImageCollection:
    baseline = lst_monthly.filterDate(baseline_start, baseline_end)
    def _tci(img):
        month = ee.Number(img.get("month"))
        b = baseline.filter(ee.Filter.calendarRange(month, month, "month"))
        lst_min = b.min().rename("LST_min")
        lst_max = b.max().rename("LST_max")
        lst = img.select("LST_C")
        denom = lst_max.subtract(lst_min)
        tci = lst_max.subtract(lst).divide(denom).multiply(100).rename("TCI")
        tci = tci.where(denom.eq(0), 0)
        return tci.set("system:time_start", img.get("system:time_start")).set("month", month)
    return lst_monthly.map(_tci)

def compute_vhi(vci_coll: ee.ImageCollection, tci_coll: ee.ImageCollection, alpha: float=0.5) -> ee.ImageCollection:
    join = ee.ImageCollection(ee.Join.inner().apply(
        vci_coll, tci_coll, ee.Filter.equals(leftField="system:time_start", rightField="system:time_start")
    ))
    def _combine(pair):
        vci = ee.Image(ee.Dictionary(pair).get("primary"))
        tci = ee.Image(ee.Dictionary(pair).get("secondary"))
        vhi = vci.multiply(alpha).add(tci.multiply(1-alpha)).rename("VHI")
        return vhi.set("system:time_start", vci.get("system:time_start")).set("month", vci.get("month"))
    return join.map(_combine)

def classify_vhi(vhi_img: ee.Image) -> ee.Image:
    v = vhi_img
    klass = ee.Image(0) \
        .where(v.lt(10), 1) \
        .where(v.gte(10).And(v.lt(20)), 2) \
        .where(v.gte(20).And(v.lt(30)), 3) \
        .where(v.gte(30).And(v.lt(40)), 4) \
        .where(v.gte(40), 5)
    return klass.rename("VHI_class").toUint8()

def monthly_chirps_collection(start: str, end: str, roi: ee.Geometry) -> ee.ImageCollection:
    # precip in mm/month band name: 'precipitation'
    return (ee.ImageCollection(CHIRPS_MO_ID)
            .filterDate(start, end)
            .filterBounds(roi)
            .select("precipitation"))

# -----------------------
# SPI computation (z-score approximation) for SPI-3
#   1) Monthly precip -> rolling 3-month sum
#   2) Baseline per calendar month: mean & std of 3-mo sums
#   3) SPI = (sum3 - mean_baseline) / std_baseline
# -----------------------
def rolling_sum_3mo(ic: ee.ImageCollection) -> ee.ImageCollection:
    # build a sequence over months, and for each, sum [t-2, t] months
    start = ee.Date(ee.Image(ic.first()).get("system:time_start"))
    end   = ee.Date(ee.Image(ic.sort("system:time_start", False).first()).get("system:time_start"))
    nmo = end.difference(start, "month").round()
    seq = ee.List.sequence(0, nmo)

    def _sum_at(i):
        i = ee.Number(i)
        t0 = start.advance(i, "month")
        t1 = t0.advance(1, "month")
        sum3 = ic.filterDate(t0.advance(-2, "month"), t1).sum().rename("P3")
        return sum3.set("system:time_start", t0.millis()).set("month", t0.get("month"))
    return ee.ImageCollection(seq.map(_sum_at))

def compute_spi3_zscore(precip_monthly: ee.ImageCollection, baseline_start: str=DEFAULT_BASELINE[0], baseline_end: str=DEFAULT_BASELINE[1]) -> ee.ImageCollection:
    p3_all = rolling_sum_3mo(precip_monthly)  # all months (baseline + analysis)
    baseline = p3_all.filterDate(baseline_start, baseline_end)

    def _spi(img):
        m = ee.Number(img.get("month"))
        clim = baseline.filter(ee.Filter.calendarRange(m, m, "month"))
        mu = clim.mean().rename("mu")
        sd = clim.reduce(ee.Reducer.stdDev()).rename("sd")
        spi = img.select("P3").subtract(mu).divide(sd).rename("SPI3")
        # guard for sd=0
        spi = spi.where(sd.eq(0), 0)
        return spi.set("system:time_start", img.get("system:time_start")).set("month", m)

    return p3_all.map(_spi)

# -----------------------
# TIME SERIES (robust)
#   (ported from your app; mean over ROI, server-side reduce, small ROI only)
# -----------------------
def timeseries_mean_robust(ic: ee.ImageCollection, band: str, roi: ee.Geometry, base_scale_m: int):
    roi_s = simplify_roi_for_stats(roi, base_scale_m)
    scale = pick_working_scale(roi_s, base_scale_m)

    def _to_feature(img):
        stats = img.select(band).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi_s,
            scale=scale,
            bestEffort=True,
            maxPixels=1e13,
            tileScale=16,
        )
        return ee.Feature(None, {
            "t": ee.Date(img.get("system:time_start")).millis(),
            "v": stats.get(band),
        })

    fc = ee.FeatureCollection(ic.map(_to_feature)).filter(ee.Filter.notNull(["v"]))
    lst   = fc.toList(fc.size())
    times = ee.List(lst.map(lambda f: ee.Feature(f).get("t"))).getInfo()
    vals  = ee.List(lst.map(lambda f: ee.Feature(f).get("v"))).getInfo()

    out = []
    for t, v in zip(times, vals):
        if t is None or v is None:
            continue
        dt = datetime.date.fromtimestamp(int(t) / 1000)
        out.append((dt, float(v)))
    out.sort(key=lambda x: x[0])
    return out

# -----------------------
# MAP helpers (EE -> ipyleaflet)
# -----------------------
def ee_tile_layer(img: ee.Image, vis: Dict, name: str) -> TileLayer:
    m = img.getMapId(vis)
    return TileLayer(url=m["tile_fetcher"].url_format, name=name, opacity=1.0)

def vis_image_for_map(img: ee.Image, vis: Dict, roi: ee.Geometry, display_scale_m: int, categorical: bool=False) -> ee.Image:
    roi_disp = simplify_roi_for_stats(roi, display_scale_m)
    
    # --- FIX 2a: Transform ROI to image's native projection *before* clipping ---
    # This avoids the "Unable to transform edge" error by performing the clip
    # in the image's native projection (e.g., SR-ORG:6974 for MODIS).
    try:
        img_proj = img.projection()
        roi_native = roi_disp.transform(img_proj, maxError=display_scale_m)
        base = img.clip(roi_native)
    except Exception as e:
        # Fallback in case transform fails
        base = img.clip(roi_disp)
    # --- END FIX ---
    
    base = base.resample('nearest' if categorical else 'bilinear')
    rgb  = base.unmask(0).visualize(**vis)
    
    # We reproject the final RGB image to EPSG:4326 so it can be displayed as a tile on the map.
    return rgb.reproject(ee.Projection('EPSG:4326').atScale(display_scale_m))

def tile_or_static_layer(img: ee.Image, vis: Dict, roi: ee.Geometry, disp_scale: int, name: str, categorical=False):
    # try coarse reprojected tile; fallback to static overlay
    try:
        disp = vis_image_for_map(img, vis, roi, disp_scale, categorical)
        return ee_tile_layer(disp, {}, name)
    except Exception:
        # static PNG overlay
        bounds = ee.Geometry(roi).bounds(1)
        coords = bounds.coordinates().get(0).getInfo()
        lons = [c[0] for c in coords]; lats = [c[1] for c in coords]
        west, east = min(lons), max(lons); south, north = min(lats), max(lats)
        rgb = vis_image_for_map(img, vis, roi, int(disp_scale*8), True)
        url = rgb.getThumbURL({"region": roi, "dimensions": 1280, "crs":"EPSG:4326"})
        return ImageOverlay(url=url, bounds=((south, west), (north, east)), name=f"{name} (static)")

def bounds_from_roi(roi: ee.Geometry):
    b = ee.Geometry(roi).bounds(1).coordinates().get(0).getInfo()
    lons = [c[0] for c in b]; lats = [c[1] for c in b]
    return (min(lats), min(lons)), (max(lats), max(lons))  # (south,west),(north,east)

def roi_bounds_3857(roi: ee.Geometry):
    # returns (xmin, ymin, xmax, ymax) in EPSG:3857
    sw, ne = bounds_from_roi(roi)
    transformer = Transformer.from_crs(4326, 3857, always_xy=True)
    xmin, ymin = transformer.transform(sw[1], sw[0])
    xmax, ymax = transformer.transform(ne[1], ne[0])
    return xmin, ymin, xmax, ymax

def asis_export_image_url(service_url: str, roi: ee.Geometry, width: int, height: int, when: datetime.date, monthly: bool=True):
    xmin, ymin, xmax, ymax = roi_bounds_3857(roi)
    params = {
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR": 102100,
        "imageSR": 3857,
        "size": f"{width},{height}",
        "format": "png",
        "f": "image",
        "renderingRule": json.dumps({"rasterFunction":"None"}),
        "interpolation": "RSP_NearestNeighbor",
    }
    if monthly:
        params["mosaicRule"] = json.dumps({
            "mosaicMethod":"esriMosaicLockRaster",
            "where": f"Year = {when.year} AND Month = {when.month}",
        })
    else:
        # dekadal not implemented here
        pass
    qs = "&".join([f"{k}={requests.utils.quote(str(v))}" for k,v in params.items()])
    return f"{service_url}/exportImage?{qs}"

# -----------------------
# SOLARA STATE
# -----------------------
roi_mode = sl.reactive("Administrative boundary")  # or "Upload boundary"
admin_level = sl.reactive("province")
province = sl.reactive(None)
district = sl.reactive(None)

# Only analysis dates remain visible in the UI
analysis_start = sl.reactive(DEFAULT_ANALYSIS[0])
analysis_end   = sl.reactive(DEFAULT_ANALYSIS[1])

basemap_choice = sl.reactive("SATELLITE")  # "TERRAIN", "ESA WorldCover 2021"
data_layer     = sl.reactive("VHI (%)")
include_ndvi   = sl.reactive(True)
data_source    = sl.reactive("GEE")  # "GEE" | "ASIS"

run_clicked    = sl.reactive(False)

_uploaded_files = sl.reactive([])  # list of {"name":..,"data":..}

# computed
roi_geom = sl.reactive(None)     # ee.Geometry
region_label = sl.reactive("")
roi_area_km2 = sl.reactive(None)
disp_scale = sl.reactive(3000)

# results
last_spi_img = sl.reactive(None)   # ee.Image
last_ndvi_img= sl.reactive(None)
last_vci_img = sl.reactive(None)
last_tci_img = sl.reactive(None)
last_vhi_img = sl.reactive(None)
last_vhic_img= sl.reactive(None)
ts_spi  = sl.reactive([])          # list[(date,val)]
ts_ndvi = sl.reactive([])
ts_vci  = sl.reactive([])
ts_tci  = sl.reactive([])
ts_vhi  = sl.reactive([])
include_spi_in_pdf = sl.reactive(True)

# -----------------------
# MAIN PAGE
# -----------------------
@sl.component
def Sidebar():
    with sl.Card("Administrative/User defined boundary"):
        sl.ToggleButtonsSingle(value=roi_mode, values=["Administrative boundary", "Upload boundary"])
    if roi_mode.value == "Administrative boundary":
        with sl.Card("Choose Administrative level"):
            sl.Select(label="Admin level", value=admin_level, values=["province","district","country"])
            if admin_level.value in ("province","district"):
                ps = list_provinces()
                # preselect Gauteng if present, else first
                default = "Gauteng" if "Gauteng" in ps else ps[0]
                if province.value is None:
                    province.set(default)
                sl.Select(label="Province", value=province, values=ps)
            if admin_level.value == "district" and province.value:
                ds = list_districts(province.value)
                if district.value is None and ds:
                    district.set(ds[0])
                sl.Select(label="District", value=district, values=ds if ds else [])
    else:
        with sl.Card("Upload ROI"):
            sl.FileDrop(label="Drop .zip (shp), .geojson, .kml", on_file=lambda f: _uploaded_files.value.append({"name": f["name"], "data": f["data"]}))
            if _uploaded_files.value:
                sl.Markdown(f"**Files:** {', '.join([f['name'] for f in _uploaded_files.value])}")
            if sl.Button("Clear uploads"):
                _uploaded_files.set([])

    with sl.Card("Base Map & Layers"):
        sl.Select(label="Basemap", value=basemap_choice, values=["SATELLITE", "TERRAIN", "ESA WorldCover 2021"])
        sl.Select(label="Data source", value=data_source, values=["GEE","ASIS"])
        if data_source.value == "GEE":
            sl.Select(label="Data layer", value=data_layer, values=["NDVI (0–1)", "VCI (%)", "TCI (%)", "VHI (%)", "VHI classification", "SPI-3 (z-score)"])
        else:
            sl.Select(label="Data layer", value=data_layer, values=["VCI (ASIS monthly)", "VHI (ASIS monthly)"])
        sl.Switch(label="Also compute NDVI layer + chart (ROI < 25 km2 only)", value=include_ndvi)

    with sl.Card("Dates (YYYY-MM-DD)"):
        sl.InputText("Analysis start", value=analysis_start)
        sl.InputText("Analysis end", value=analysis_end)
        
    # The '5) Computation settings' card has been removed as requested.

    with sl.Card():
        if sl.Button("Run analysis"):
            run_clicked.set(True)

    # info
    if roi_area_km2.value is not None:
        sl.Markdown(f"**ROI area** ≈ {roi_area_km2.value:,.2f} km² • **display scale**: {disp_scale.value} m")


@sl.component
def MapPanel():
    # Build ROI
    if roi_mode.value == "Administrative boundary":
        try:
            g, label = get_roi_admin(admin_level.value, province.value, district.value)
        except Exception as e:
            sl.Error(f"Admin selection incomplete: {e}")
            return
    else:
        if not _uploaded_files.value:
            sl.Info("Upload a boundary to proceed.")
            return
        try:
            g, label = roi_from_uploaded(_uploaded_files.value)
        except Exception as e:
            sl.Error(f"Upload error: {e}")
            return

    roi_geom.set(g); region_label.set(label)
    area_km2 = ee.Number(g.area(1)).divide(1e6).getInfo()
    roi_area_km2.set(area_km2)
    disp_scale.set(pick_display_scale(area_km2))

    # Initialize ipyleaflet map
    # Basemap
    m = LeafletMap(zoom=5, scroll_wheel_zoom=True, center=(-28.5, 24.7))  # SA approx center
    if   basemap_choice.value == "SATELLITE":
        m.add_layer(basemap_to_tiles(basemaps.Esri.WorldImagery))
    elif basemap_choice.value == "TERRAIN":
        m.add_layer(basemap_to_tiles(basemaps.Esri.WorldTerrain))
    elif basemap_choice.value == "ESA WorldCover 2021":
        try:
            # Render ESA WorldCover 2021 from EE as a base layer
            wc = ee.Image("ESA/WorldCover/v200").select("Map")
            # Official palette ordering for classes
            wc_palette = [
                "006400","ffbb22","ffff4c","f096ff","fa0000",
                "b4b4b4","f0f0f0","0064c8","0096a0","00cf75","fae6a0"
            ]
            wc_vis = {"min":1, "max":100, "palette": wc_palette}
            layer = tile_or_static_layer(wc, wc_vis, g, disp_scale.value, "ESA WorldCover 2021", categorical=True)
            m.add_layer(layer)
        except Exception:
            # fallback to OSM if EE layer fails
            m.add_layer(basemap_to_tiles(basemaps.OpenStreetMap.Mapnik))
    else:
        m.add_layer(basemap_to_tiles(basemaps.OpenStreetMap.Mapnik))
    m.add_control(LayersControl(position="topright"))

    # ROI outline as static tile (cheap): use getThumbURL
    try:
        bounds = ee.Geometry(g).bounds(1)
        rgb = ee.Image().paint(ee.FeatureCollection(ee.Feature(g)), 0, 2).visualize(palette=["yellow"])
        url = rgb.getThumbURL({"region": g, "dimensions": 1024, "crs": "EPSG:4326"})
        coords = bounds.coordinates().get(0).getInfo()
        lons = [c[0] for c in coords]; lats = [c[1] for c in coords]
        overlay = ImageOverlay(url=url, bounds=((min(lats), min(lons)), (max(lats), max(lons))), name="ROI")
        m.add_layer(overlay)
    except Exception:
        pass

    # center to ROI
    try:
        sw, ne = bounds_from_roi(g)
        m.fit_bounds([sw, ne])
    except Exception:
        pass

    # RUN COMPUTATIONS
    if run_clicked.value:
        with sl.Card("Status"):
            sl.Info("Computing indices…")
        
        # Use hardcoded baseline dates from constants
        base_s = DEFAULT_BASELINE[0]
        base_e = DEFAULT_BASELINE[1]
        start_a = str(analysis_start.value)
        end_a   = str(analysis_end.value)

        # Build collections (analysis + baseline where needed)
        ndvi_img = None
        # NDVI analysis collection
        ndvi_m_an = monthly_ndvi_collection(start_a, end_a, g)
        if include_ndvi.value:
            ndvi_img = ee.Image(ndvi_m_an.sort("system:time_start", False).first()).clip(g).select("NDVI")
            last_ndvi_img.set(ndvi_img)

        # LST monthly (for TCI) analysis collection
        lst_m_an = monthly_lstC_collection(start_a, end_a, g)
        
        # Baseline-augmented collections (used for VCI/TCI/VHI computation)
        # Note: Functions now default to using DEFAULT_BASELINE constants.
        ndvi_m_all = monthly_ndvi_collection(base_s, base_e, g).merge(ndvi_m_an)
        lst_m_all  = monthly_lstC_collection(base_s, base_e, g).merge(lst_m_an)

        # Compute VCI/TCI/VHI over analysis window
        vci_m = compute_vci(ndvi_m_all).filterDate(start_a, end_a)
        tci_m = compute_tci(lst_m_all).filterDate(start_a, end_a)
        vhi_m = compute_vhi(vci_m, tci_m, alpha=0.5)

        # Extract last images
        def _last(ic):
            return ee.Image(ic.sort("system:time_start", False).first()).clip(g)
        try:
            last_vci_img.set(_last(vci_m).select("VCI"))
        except Exception:
            pass
        try:
            last_tci_img.set(_last(tci_m).select("TCI"))
        except Exception:
            pass
        try:
            _vhi = _last(vhi_m).select("VHI")
            last_vhi_img.set(_vhi)
            last_vhic_img.set(classify_vhi(_vhi))
        except Exception:
            pass

        # SPI-3 (z-score)
        chirps_all = monthly_chirps_collection(base_s, end_a, g)  # include baseline and analysis
        spi3_all   = compute_spi3_zscore(chirps_all) # Function uses DEFAULT_BASELINE
        spi3_an    = spi3_all.filterDate(start_a, end_a)
        try:
            spi_img = ee.Image(spi3_an.sort("system:time_start", False).first()).clip(g).select("SPI3")
            last_spi_img.set(spi_img)
        except Exception:
            sl.Warning("No CHIRPS data available for the selected window.")
            return

        # Map layer based on selection
        VIS_SPI = {"min": -2.5, "max": 2.5, "palette": PALETTE_SPI}
        VIS_NDVI = {"min":0.0, "max":1.0,   "palette": PALETTE_NDVI}
        VIS_PERC = {"min":0.0, "max":100.0, "palette": PALETTE_PERC}
        VIS_VHIC = {"min":1,   "max":5,     "palette": PALETTE_VHI_CLASS}

        try:
            if data_source.value == "GEE":
                if data_layer.value == "SPI-3 (z-score)":
                    layer = tile_or_static_layer(spi_img, VIS_SPI, g, disp_scale.value, "SPI-3", categorical=False)
                    m.add_layer(layer)
                elif data_layer.value == "NDVI (0–1)" and last_ndvi_img.value is not None:
                    layer = tile_or_static_layer(last_ndvi_img.value, VIS_NDVI, g, disp_scale.value, "NDVI", categorical=False)
                    m.add_layer(layer)
                elif data_layer.value == "VCI (%)" and last_vci_img.value is not None:
                    layer = tile_or_static_layer(last_vci_img.value, VIS_PERC, g, disp_scale.value, "VCI", categorical=False)
                    m.add_layer(layer)
                elif data_layer.value == "TCI (%)" and last_tci_img.value is not None:
                    layer = tile_or_static_layer(last_tci_img.value, VIS_PERC, g, disp_scale.value, "TCI", categorical=False)
                    m.add_layer(layer)
                elif data_layer.value == "VHI (%)" and last_vhi_img.value is not None:
                    layer = tile_or_static_layer(last_vhi_img.value, VIS_PERC, g, disp_scale.value, "VHI", categorical=False)
                    m.add_layer(layer)
                elif data_layer.value == "VHI classification" and last_vhic_img.value is not None:
                    layer = tile_or_static_layer(last_vhic_img.value, VIS_VHIC, g, disp_scale.value, "VHI class", categorical=True)
                    m.add_layer(layer)
            else:
                # ASIS overlays (monthly by analysis_end)
                when = pd.to_datetime(analysis_end.value).date()
                svc = ASIS_VCI_M if "VCI" in data_layer.value else ASIS_VHI_M
                url = asis_export_image_url(svc, g, width=1024, height=1024, when=when, monthly=True)
                # bounds in lat/lon
                sw, ne = bounds_from_roi(g)
                overlay = ImageOverlay(url=url, bounds=(sw, ne), name=data_layer.value)
                m.add_layer(overlay)
        except Exception as e:
            sl.Warning(f"Could not render tiles; fallback used where possible: {e}")

        # Optional NDVI layer
        if ndvi_img is not None:
            VIS_NDVI = {"min":0.0, "max":1.0, "palette": PALETTE_NDVI}
            try:
                layer_ndvi = tile_or_static_layer(ndvi_img, VIS_NDVI, g, disp_scale.value, "NDVI", categorical=False)
                m.add_layer(layer_ndvi)
            except Exception:
                pass

        # Charts (only for small ROI)
        if area_km2 <= KM2_THRESHOLD:
            # Use hardcoded scale constants (NDVI_SCALE_M, SPI_SCALE_M, LST_SCALE_M)
            try:
                ts_spi.set(timeseries_mean_robust(spi3_an, "SPI3", g, SPI_SCALE_M))
            except Exception:
                pass
            try:
                if include_ndvi.value:
                    ts_ndvi.set(timeseries_mean_robust(ndvi_m_an, "NDVI", g, NDVI_SCALE_M))
            except Exception:
                pass
            try:
                ts_vci.set(timeseries_mean_robust(vci_m, "VCI", g, NDVI_SCALE_M))
            except Exception:
                pass
            try:
                ts_tci.set(timeseries_mean_robust(tci_m, "TCI", g, LST_SCALE_M))
            except Exception:
                pass
            try:
                ts_vhi.set(timeseries_mean_robust(vhi_m, "VHI", g, NDVI_SCALE_M))
            except Exception:
                pass
        else:
            sl.Info("ROI > 2,500 ha: Only map classifications are shown to avoid EE memory limits.")

    # FIX 1: Replaced w.Widget(widget=m) with sl.Element(m) for Solara compatibility
    sl.Element(m)

@sl.component
def ChartsAndExport():
    small = (roi_area_km2.value is not None) and (roi_area_km2.value <= KM2_THRESHOLD)

    with sl.Card("Time-series"):
        if not run_clicked.value:
            sl.Info("Click **Run analysis** to compute charts.")
        elif not small:
            sl.Warning(f"ROI area is {roi_area_km2.value:,.2f} km² (> 25 km²). Charts disabled.")
        else:
            # Simple SPI line chart
            if ts_spi.value:
                df_spi = pd.DataFrame(ts_spi.value, columns=["date","SPI3"]).set_index("date")
                sl.LinePlot(df_spi, title=f"SPI-3 • {region_label.value}")
            if include_ndvi.value and ts_ndvi.value:
                df_ndvi = pd.DataFrame(ts_ndvi.value, columns=["date","NDVI"]).set_index("date")
                sl.LinePlot(df_ndvi, title=f"NDVI • {region_label.value}")
            if ts_vci.value:
                df_vci = pd.DataFrame(ts_vci.value, columns=["date","VCI"]).set_index("date")
                sl.LinePlot(df_vci, title=f"VCI • {region_label.value}")
            if ts_tci.value:
                df_tci = pd.DataFrame(ts_tci.value, columns=["date","TCI"]).set_index("date")
                sl.LinePlot(df_tci, title=f"TCI • {region_label.value}")
            if ts_vhi.value:
                df_vhi = pd.DataFrame(ts_vhi.value, columns=["date","VHI"]).set_index("date")
                sl.LinePlot(df_vhi, title=f"VHI • {region_label.value}")

    with sl.Card("Bulletin (PDF)"):
        sl.Markdown("Generates a bulletin with NDVI/VCI/TCI/VHI maps and charts (if small ROI). Optionally include SPI.")
        sl.Switch(label="Include SPI (map and chart) in PDF", value=include_spi_in_pdf)
        if sl.Button("Build PDF"):
            # prepare map PNGs via EE thumbnails
            g = roi_geom.value
            if g is None:
                sl.Warning("Run analysis first.")
                return
            dscale = disp_scale.value
            VIS_NDVI = {"min":0.0, "max":1.0,   "palette": PALETTE_NDVI}
            VIS_PERC = {"min":0.0, "max":100.0, "palette": PALETTE_PERC}
            VIS_VHIC = {"min":1,   "max":5,     "palette": PALETTE_VHI_CLASS}
            VIS_SPI  = {"min": -2.5, "max": 2.5, "palette": PALETTE_SPI}

            ndvi_png = image_png_bytes(last_ndvi_img.value, g, VIS_NDVI, dscale, dimensions=900, categorical=False) if (data_source.value=="GEE" and last_ndvi_img.value is not None) else None
            vci_png  = None
            tci_png  = None
            vhi_png  = None
            if data_source.value == "GEE":
                vci_png  = image_png_bytes(last_vci_img.value,  g, VIS_PERC, dscale, dimensions=900, categorical=False) if last_vci_img.value is not None else None
                tci_png  = image_png_bytes(last_tci_img.value,  g, VIS_PERC, dscale, dimensions=900, categorical=False) if last_tci_img.value is not None else None
                vhi_png  = image_png_bytes(last_vhi_img.value,  g, VIS_PERC, dscale, dimensions=900, categorical=False) if last_vhi_img.value is not None else None
            else:
                when = pd.to_datetime(analysis_end.value).date()
                # ASIS monthly VCI/VHI panels
                try:
                    url_vci = asis_export_image_url(ASIS_VCI_M, g, 1024, 1024, when, monthly=True)
                    sw, ne = bounds_from_roi(g)
                    # fetch the image directly
                    vci_png = requests.get(url_vci, timeout=180).content
                except Exception:
                    vci_png = None
                try:
                    url_vhi = asis_export_image_url(ASIS_VHI_M, g, 1024, 1024, when, monthly=True)
                    vhi_png = requests.get(url_vhi, timeout=180).content
                except Exception:
                    vhi_png = None
            vhic_png = image_png_bytes(last_vhic_img.value, g, VIS_VHIC, dscale, dimensions=900, categorical=True)  if last_vhic_img.value is not None else None
            spi_png  = image_png_bytes(last_spi_img.value,  g, VIS_SPI,  dscale, dimensions=900, categorical=False) if include_spi_in_pdf.value and last_spi_img.value is not None else None

            # charts (if small)
            charts = {}
            if small:
                def _chart_png(series, title, ylab):
                    s = pd.Series({d:v for d,v in series})
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(8,3))
                    ax.plot(s.index, s.values, marker='o'); ax.grid(True, alpha=0.3)
                    ax.set_title(title); ax.set_xlabel("Date"); ax.set_ylabel(ylab)
                    fig.tight_layout()
                    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=200)
                    plt.close(fig); return buf.getvalue()
                if ts_ndvi.value:
                    charts["NDVI"] = _chart_png(ts_ndvi.value, f"NDVI • {region_label.value}", "NDVI")
                if ts_vci.value:
                    charts["VCI"]  = _chart_png(ts_vci.value,  f"VCI • {region_label.value}",  "VCI")
                if ts_tci.value:
                    charts["TCI"]  = _chart_png(ts_tci.value,  f"TCI • {region_label.value}",  "TCI")
                if ts_vhi.value:
                    charts["VHI"]  = _chart_png(ts_vhi.value,  f"VHI • {region_label.value}",  "VHI")
                if include_spi_in_pdf.value and ts_spi.value:
                    charts["SPI3"] = _chart_png(ts_spi.value,  f"SPI-3 • {region_label.value}",  "SPI-3")

            # write temp files
            tmpdir = Path(tempfile.mkdtemp())
            p_ndvi = tmpdir/"map_ndvi.png" if ndvi_png else None
            if p_ndvi: p_ndvi.write_bytes(ndvi_png)
            p_vci  = tmpdir/"map_vci.png"  if vci_png  else None
            if p_vci:  p_vci.write_bytes(vci_png)
            p_tci  = tmpdir/"map_tci.png"  if tci_png  else None
            if p_tci:  p_tci.write_bytes(tci_png)
            p_vhi  = tmpdir/"map_vhi.png"  if vhi_png  else None
            if p_vhi:  p_vhi.write_bytes(vhi_png)
            p_vhic = tmpdir/"map_vhi_class.png" if vhic_png else None
            if p_vhic: p_vhic.write_bytes(vhic_png)
            p_spi  = tmpdir/"map_spi.png" if spi_png else None
            if p_spi:  p_spi.write_bytes(spi_png)

            # build PDF
            pdf = FPDF(orientation="P", unit="mm", format="A4")
            pdf.set_auto_page_break(auto=True, margin=12)
            pdf.add_page()
            pdf.set_font("Arial", "B", 16); pdf.cell(0, 10, "South Africa Drought Monitoring Bulletin", ln=1, align="C")
            pdf.set_font("Arial", "", 11)
            today = datetime.date.today().strftime("%Y-%m-%d")
            pdf.cell(0, 7, f"{region_label.value} - Generated {today}", ln=1, align="C")
            pdf.ln(4)

            colw, rowh = 95, 60
            x0, y0 = 10, pdf.get_y()
            def slot(x, y, label, path):
                pdf.set_xy(x, y); pdf.set_font("Arial","B",11); pdf.cell(colw-10, 6, label, ln=1)
                pdf.image(str(path), x=x, y=y+6, w=colw-10, h=rowh)

            if p_ndvi:
                slot(x0 + 0*colw, y0 + 0*(rowh+10), "NDVI", p_ndvi)
            if p_vci:
                slot(x0 + 1*colw, y0 + 0*(rowh+10), "VCI (%)", p_vci)
            if p_tci:
                slot(x0 + 0*colw, y0 + 1*(rowh+10), "TCI (%)", p_tci)
            if p_vhi:
                slot(x0 + 1*colw, y0 + 1*(rowh+10), "VHI (%)", p_vhi)
            if p_vhic:
                slot(x0 + 0*colw, y0 + 2*(rowh+10), "VHI class", p_vhic)
            if p_spi:
                slot(x0 + 1*colw, y0 + 2*(rowh+10), "SPI-3 (z-score)", p_spi)

            if small and charts:
                pdf.add_page(); pdf.set_font("Arial","B",14); pdf.cell(0,8,"Zonal mean time-series",ln=1)
                for label in ["NDVI","VCI","TCI","VHI","SPI3"]:
                    if label in charts:
                        pdf.set_font("Arial","B",12); pdf.cell(0,7,label,ln=1)
                        p = tmpdir / f"ts_{label.lower()}.png"
                        p.write_bytes(charts[label])
                        pdf.image(str(p), x=10, y=pdf.get_y(), w=190); pdf.ln(67)

            pdf.add_page(); pdf.set_font("Arial","B",14); pdf.cell(0,8,"Brief interpretation",ln=1)
            pdf.set_font("Arial","",11)
            bullet = (
                "- NDVI shows vegetation greenness; sustained lows may indicate stress.\n"
                "- VCI/TCI (0-100): <40 often suggests drought stress.\n"
                "- VHI combines both; persistent lows imply more severe conditions.\n"
                f"- Note: GEE calculations use a fixed baseline period: **{DEFAULT_BASELINE[0]}** to **{DEFAULT_BASELINE[1]}**.\n"
            )
            if not small:
                bullet += "- Note: ROI exceeds 2,500 ha; time-series omitted to avoid EE memory limits."
            pdf.multi_cell(0,6, bullet)
            pdf_bytes = pdf.output(dest="S").encode("latin1")
            fn = f"SA_Drought_Bulletin_{region_label.value.replace(': ','_').replace(',','_').replace(' ','_')}.pdf"
            href = f"data:application/pdf;base64,{base64.b64encode(pdf_bytes).decode('ascii')}"
            sl.Markdown(f"[⬇️ Download PDF ({fn})]({href})")

# -----------------------
# SMALL HELPERS reused from your app (PNG thumbnail creation)
# -----------------------
def image_png_bytes(img: ee.Image, roi: ee.Geometry, vis: Dict, display_scale_m: int, dimensions=900, categorical: bool=False) -> bytes:
    roi_disp = simplify_roi_for_stats(roi, display_scale_m)

    # --- FIX 2b: Transform ROI to image's native projection *before* clipping ---
    # This avoids the "Unable to transform edge" error.
    try:
        img_proj = img.projection()
        roi_native = roi_disp.transform(img_proj, maxError=display_scale_m)
        base = img.clip(roi_native)
    except Exception as e:
        base = img.clip(roi_disp)
    # --- END FIX ---
    
    base = base.resample('nearest' if categorical else 'bilinear')
    rgb = base.unmask(0).visualize(**vis)
    
    # getThumbURL handles the final reprojection to crs: "EPSG:4326"
    url = rgb.getThumbURL({
        "region": roi_disp,
        "dimensions": dimensions,
        "crs": "EPSG:4326"
    })
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    return r.content

# -----------------------
# PAGE LAYOUT
# -----------------------
@sl.component
def Page():
    with sl.Sidebar():
        Sidebar()
    sl.Markdown("### Vegetation and Precipitation Indices based Drought Conditions and Severity Monitoring. This system uses Vegetation Health Index (VHI) derived from MODIS NDVI and LST, and Standardized Precipitation Index (SPI) from CHIRPS precipitation data.")
    sl.Columns([2, 1], [MapPanel(), ChartsAndExport()]) # Use Columns for better layout