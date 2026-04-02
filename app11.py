# app.py
# Black Spot Analysis Dashboard (show ALL black spots with circles)
# Black-spot rule: for every point, form a 250 m circle; if >= threshold points inside -> black spot
# Fixed: always display/select stripped sheet names (no label-shortening). This fixes missing-sheet issues like "202".
# Author: You + GPT-5 Thinking

import io
import re
import html
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import DBSCAN
import folium 
from folium import Circle, CircleMarker, FeatureGroup, Popup, IFrame, Element, Circle as FoliumCircle
from streamlit_folium import st_folium

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Black Spot Analysis", layout="wide", initial_sidebar_state="expanded")
# Default path (uploaded file)
DEFAULT_DATA_PATH = "/Users/adityaacharya/Desktop/ITS/NH-53 Accident.xlsx"
BUFFER_METERS = 250                      # circle radius for each point (meters)
EARTH_RADIUS_M = 6_371_000               # Earth radius in meters
DEFAULT_BLACKSPOT_THRESHOLD = 3          # default points required inside a circle to mark black spot

st.markdown(
    """
    <style>
    .main .block-container { padding-top: 1rem; padding-bottom: 1rem; padding-left: 2rem; padding-right: 2rem; }
    iframe[title="streamlit_folium"] { margin-left: auto; margin-right: auto; display: block; }
    .small-note { font-size:12px; color:#666; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Utilities (parsing + robust sheet read)
# ----------------------------
HEADER_TOKENS = [
    "latitude", "lat", "longitude", "lon", "long",
    "date", "accident date", "time", "accident time",
    "location", "place", "site", "remarks", "remark", "description",
    "accident location", "acc_location"
]

def try_guess_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    cols_norm = {c: str(c).lower().replace("\n", " ") for c in cols}
    for cand in candidates:
        key = cand.lower()
        for raw, norm in cols_norm.items():
            if key in norm:
                return raw
    return None

def find_col_by_cells(df: pd.DataFrame, candidates: List[str], sample_rows: int = 40) -> Optional[str]:
    tokens = [c.lower() for c in candidates]
    for col in df.columns:
        try:
            ser = df[col].astype(str).str.lower().head(sample_rows)
        except Exception:
            continue
        text = " ".join(ser)
        for token in tokens:
            if token in text:
                return col
    return None

def dms_to_decimal(s: str) -> float:
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    if not s:
        return np.nan
    s = s.replace("″", "''").replace("’", "'").replace("“", "").replace("”", "")
    m = re.match(r"^\s*([NSEWnsew])\s*(\d+)[°:\s]\s*(\d+)[\'’]\s*(\d+(?:\.\d*)?)", s)
    if not m:
        return np.nan
    hemi = m.group(1).upper()
    deg = float(m.group(2))
    minutes = float(m.group(3))
    seconds = float(m.group(4))
    dec = deg + minutes / 60 + seconds / 3600
    if hemi in ("S", "W"):
        dec = -dec
    return dec

def parse_coord_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    sample = series.dropna().astype(str).head(30).str.upper()
    if sample.str.contains("N").any() or sample.str.contains("S").any() or sample.str.contains("E").any() or sample.str.contains("W").any():
        return series.apply(dms_to_decimal)
    return pd.to_numeric(series, errors="coerce")

# robust header detection & unique header builder
def detect_header_row(df_raw: pd.DataFrame, max_rows: int = 15) -> Optional[int]:
    rows = min(max_rows, len(df_raw))
    for r in range(rows):
        row_text = " ".join([str(x).lower() for x in df_raw.iloc[r].values])
        score = sum(1 for token in HEADER_TOKENS if token in row_text)
        if score >= 1:
            return r
    return None

def make_unique_headers(headers: List[str]) -> List[str]:
    cleaned = [("" if (h is None or str(h).strip() == "") else str(h).strip()) for h in headers]
    out = []
    counts = {}
    for i, h in enumerate(cleaned):
        base = f"col_{i}" if h == "" else h
        if base in counts:
            counts[base] += 1
            new = f"{base}_{counts[base]}"
        else:
            counts[base] = 0
            new = base
        out.append(new)
    return out

def read_sheet_with_fallback(xls: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    df_raw = xls.parse(sheet_name, header=None, dtype=str)
    header_row = detect_header_row(df_raw, max_rows=15)
    if header_row is not None:
        header_vals = df_raw.iloc[header_row].fillna("").astype(str).tolist()
        data_part = df_raw.iloc[header_row + 1 :].copy().reset_index(drop=True)
        n_data_cols = data_part.shape[1]
        if len(header_vals) < n_data_cols:
            header_vals += [f"col_{i}" for i in range(len(header_vals), n_data_cols)]
        elif len(header_vals) > n_data_cols:
            header_vals = header_vals[:n_data_cols]
        header_unique = make_unique_headers(header_vals)
        data_part.columns = header_unique
        data_part = data_part.dropna(how="all").reset_index(drop=True)
        data = data_part
    else:
        data = xls.parse(sheet_name)
        data.columns = [str(c).strip() for c in data.columns]
    data.columns = [str(c).strip() for c in data.columns]
    return data

# ----------------------------
# Load workbook (all sheets) and prepare combined dataframe
# ----------------------------
def load_workbook_all_sheets(file_like_or_path: str) -> Tuple[pd.DataFrame, List[str]]:
    xls = pd.ExcelFile(file_like_or_path)
    frames = []
    cleaned_sheet_names = []
    skip_names = {"ALL", "SUMMARY", "TOTAL", "SUM"}
    for sheet in xls.sheet_names:
        name_clean = str(sheet).strip()
        if name_clean.upper() in skip_names:
            continue
        df_sheet = read_sheet_with_fallback(xls, sheet)
        # store stripped sheet name in dataframe for consistent filtering later
        df_sheet["__sheet__"] = name_clean
        frames.append(df_sheet)
        cleaned_sheet_names.append(name_clean)
    if not frames:
        raise ValueError("No usable sheets found in the workbook.")
    df = pd.concat(frames, ignore_index=True, sort=False)
    df.columns = [str(c).strip() for c in df.columns]
    # return dataframe and list of stripped sheet names (so UI selects exact stripped names)
    return df, cleaned_sheet_names

# ----------------------------
# Haversine helper (vectorized)
# ----------------------------
def haversine_meters(lat1, lon1, lat2, lon2):
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lat2)
    lon2r = np.radians(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return EARTH_RADIUS_M * c

# ----------------------------
# Black-spot logic (per your rule)
# ----------------------------
def compute_blackspots_by_local_circles(df: pd.DataFrame, radius_m: int = BUFFER_METERS, threshold: int = DEFAULT_BLACKSPOT_THRESHOLD) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.assign(is_black_candidate=False, local_count=0), pd.DataFrame()

    coords = df[["lat", "lon"]].to_numpy(dtype=float)
    lat_arr = coords[:, 0]
    lon_arr = coords[:, 1]

    # Pairwise distances matrix (n x n)
    lat1 = lat_arr[:, None]
    lon1 = lon_arr[:, None]
    lat2 = lat_arr[None, :]
    lon2 = lon_arr[None, :]

    dists = haversine_meters(lat1, lon1, lat2, lon2)  # shape (n,n)

    # Count points within radius for each center point (including itself)
    counts_within = (dists <= radius_m).sum(axis=1)  # shape (n,)

    # Mark candidates
    df = df.copy()
    df["local_count"] = counts_within
    df["is_black_candidate"] = df["local_count"] >= threshold

    # If no candidates, return
    candidates = df[df["is_black_candidate"]].copy()
    if candidates.empty:
        return df, pd.DataFrame()

    # Merge overlapping candidate circles by clustering candidate POINTS using DBSCAN (haversine metric)
    coords_cand_rad = np.radians(candidates[["lat", "lon"]].to_numpy(dtype=float))
    eps_rad = radius_m / EARTH_RADIUS_M  # centers within radius -> overlapping circles
    db = DBSCAN(eps=eps_rad, min_samples=1, metric="haversine").fit(coords_cand_rad)
    candidates["cand_cluster"] = db.labels_

    # Build summary per cand_cluster: centroid is mean of lat/lon of candidate centers
    summary_rows = []
    for cl in sorted(candidates["cand_cluster"].unique()):
        group = candidates[candidates["cand_cluster"] == cl]
        centroid_lat = group["lat"].mean()
        centroid_lon = group["lon"].mean()
        # Count how many original points fall within radius_m of this centroid
        d_to_centroid = haversine_meters(centroid_lat, centroid_lon, lat_arr, lon_arr)
        accidents = int((d_to_centroid <= radius_m).sum())
        summary_rows.append({
            "cand_cluster": cl,
            "accidents": accidents,
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon
        })

    summary = pd.DataFrame(summary_rows).sort_values("accidents", ascending=False).reset_index(drop=True)
    return df, summary

# ----------------------------
# Popup builder (strict five fields)
# ----------------------------
def build_popup_html_strict(row: pd.Series, popup_cols: Dict[str, Optional[str]]) -> str:
    def get_cell(col: Optional[str]) -> str:
        if col is None or col not in row.index:
            return "—"
        val = row.get(col)
        if pd.isna(val):
            return "—"
        if col == popup_cols.get("date") and "_date_parsed" in row.index:
            dt = row.get("_date_parsed")
            return str(pd.to_datetime(dt).date()) if not pd.isna(dt) else str(val)
        return str(val)
    date_val = get_cell(popup_cols.get("date"))
    time_val = get_cell(popup_cols.get("time"))
    location_val = get_cell(popup_cols.get("location"))
    accident_location_val = get_cell(popup_cols.get("accident_location"))
    remarks_val = get_cell(popup_cols.get("remarks"))
    lat = float(row["lat"]); lon = float(row["lon"])
    html_body = f"""
    <div style="font-family: Arial, sans-serif; width:320px;">
      <h4 style="margin:0 0 6px 0; color:#0b5394;">Accident Details</h4>
      <div style="font-size:13px; color:#222;">
        <div style="margin-bottom:6px;"><strong>Date:</strong> {html.escape(date_val)}</div>
        <div style="margin-bottom:6px;"><strong>Time:</strong> {html.escape(time_val)}</div>
        <div style="margin-bottom:6px;"><strong>Location:</strong> {html.escape(location_val)}</div>
        <div style="margin-bottom:6px;"><strong>Accident location:</strong> {html.escape(accident_location_val)}</div>
        <div style="margin-bottom:6px;"><strong>Remarks:</strong> {html.escape(remarks_val)}</div>
        <div style="margin-top:8px;">
          <div style="font-size:12px;color:#666;margin-bottom:6px;"><strong>Coordinates:</strong> {lat:.6f}, {lon:.6f}</div>
          <a href="https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}" target="_blank"
             style="display:inline-block;padding:10px 12px;background:#1a73e8;color:#fff;text-decoration:none;border-radius:6px;font-weight:700;">
             Open in Google Street View
          </a>
        </div>
      </div>
    </div>
    """
    return html_body

# ----------------------------
# Map builder
# ----------------------------
def make_map(df: pd.DataFrame, blackspot_summary: pd.DataFrame, center_lat: Optional[float]=None, center_lon: Optional[float]=None) -> folium.Map:
    if df.empty:
        center_lat, center_lon = 21.17, 72.83
    else:
        center_lat = float(df["lat"].mean()) if center_lat is None else center_lat
        center_lon = float(df["lon"].mean()) if center_lon is None else center_lon

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=11, control_scale=True, tiles="OpenStreetMap")
    popup_cols = df.attrs.get("popup_cols", {})

    # Accident Points (BLUE)
    pts = FeatureGroup(name="Accident Points", show=True)
    for _, r in df.iterrows():
        lat = float(r["lat"]); lon = float(r["lon"])
        popup_html = build_popup_html_strict(r, popup_cols)
        iframe = IFrame(popup_html, width=340, height=220)
        popup = Popup(iframe, max_width=340)
        tooltip = "Hover or click to view required details & Street View"
        CircleMarker(
            location=[lat, lon],
            radius=5,
            fill=True,
            color="#0057e7",       # blue
            fill_color="#0057e7",
            fill_opacity=0.9,
            tooltip=tooltip,
            popup=popup,
        ).add_to(pts)
    pts.add_to(fmap)

    # Black spot buffers (BLACK) from summary (show ALL)
    if blackspot_summary is not None and not blackspot_summary.empty:
        buffs = FeatureGroup(name=f"Black Spots (radius {BUFFER_METERS} m)", show=True)
        for _, c in blackspot_summary.iterrows():
            FoliumCircle(
                [c["centroid_lat"], c["centroid_lon"]],
                radius=BUFFER_METERS,
                fill=True,
                fill_opacity=0.12,
                color="black",
                fill_color="black",
                weight=2,
                tooltip=f"Accidents: {int(c['accidents'])}",
            ).add_to(buffs)
            # draw small black dot at centroid
            CircleMarker(
                location=[c["centroid_lat"], c["centroid_lon"]],
                radius=4,
                fill=True,
                color="black",
                fill_color="black",
                fill_opacity=1.0,
                tooltip=f"Black spot — Accidents: {int(c['accidents'])}"
            ).add_to(buffs)
        buffs.add_to(fmap)

    folium.LayerControl().add_to(fmap)

    # JS to bind hover popups to CircleMarkers
    js = """
    <script>
    (function(){
      function bindHoverPopups(){
        try {
          var map;
          for (var k in window) {
            if (window[k] && window[k] instanceof L.Map) { map = window[k]; break; }
          }
          if (!map) {
            if (window.map) map = window.map;
            else return;
          }
          for (var id in map._layers) {
            var layer = map._layers[id];
            if (!layer) continue;
            try {
              if (layer instanceof L.CircleMarker) {
                var popup = layer.getPopup();
                if (popup) {
                  if (!layer._hasHoverBind) {
                    layer.on('mouseover', function(e){ this.openPopup(); });
                    layer.on('mouseout', function(e){ this.closePopup(); });
                    layer._hasHoverBind = true;
                  }
                }
              }
            } catch(err) {}
          }
        } catch(e) { console.log('hover popup bind error', e); }
      }
      var tries = 0;
      var interval = setInterval(function(){ try { bindHoverPopups(); } catch(e){}; tries +=1; if (tries>8) clearInterval(interval); }, 600);
    })();
    </script>
    """
    fmap.get_root().html.add_child(Element(js))

    return fmap

# ----------------------------
# UI / App flow
# ----------------------------
st.title("🛣️ Black Spot Analysis Dashboard")
st.caption(f"Black-spot rule: circle radius = {BUFFER_METERS} m; threshold = adjustable in sidebar.")

with st.sidebar:
    st.header("Data & Controls")
    up = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
    st.markdown("Or use default uploaded workbook:")
    st.code(DEFAULT_DATA_PATH)
    threshold = st.number_input("Black-spot threshold (points inside circle)", min_value=2, max_value=100, value=DEFAULT_BLACKSPOT_THRESHOLD, step=1)
    st.markdown("---")
    st.markdown("**Column mapping (per selected sheet)**")
    st.markdown("If any popup shows `—`, choose the correct column here and the map will update.")

# Load workbook
try:
    if up is not None:
        data_all, sheet_names = load_workbook_all_sheets(io.BytesIO(up.read()))
    else:
        data_all, sheet_names = load_workbook_all_sheets(DEFAULT_DATA_PATH)
except Exception as e:
    st.error(f"Error loading workbook: {e}")
    st.stop()

# IMPORTANT: use the stripped sheet names as choices so selections match df["__sheet__"] exactly
cleaned_choices = [str(s).strip() for s in sheet_names]
# Remove duplicates while preserving order
seen = set()
unique_choices = []
for s in cleaned_choices:
    if s not in seen:
        unique_choices.append(s)
        seen.add(s)

selected_sheet = st.selectbox("Select sheet / year (exact sheet names shown)", unique_choices, index=len(unique_choices)-1)

# Subset data for the selected sheet
df_sheet = data_all[data_all["__sheet__"] == selected_sheet].copy()
if df_sheet.empty:
    st.error(f"No data found in sheet: {selected_sheet}")
    st.stop()

# Try to detect lat/lon for this sheet if not already present
cols = df_sheet.columns.tolist()
lat_col_guess = try_guess_col(cols, ["latitude", "lat"]) or find_col_by_cells(df_sheet, ["latitude", "lat"])
lon_col_guess = try_guess_col(cols, ["longitude", "lon", "long"]) or find_col_by_cells(df_sheet, ["longitude", "lon", "long"])
if lat_col_guess is None or lon_col_guess is None:
    # try last columns
    possible = cols[-6:]
    for c in possible:
        sample = df_sheet[c].astype(str).head(20).str.upper()
        # check for DMS marks or NSEW letters OR decimal-like numbers
        if sample.str.contains(r"[NSEW]").any() or sample.str.contains(r"°").any() or sample.str.match(r"^-?\d+(\.\d+)?$").any():
            if lat_col_guess is None:
                lat_col_guess = c
            elif lon_col_guess is None and c != lat_col_guess:
                lon_col_guess = c

if lat_col_guess and lon_col_guess:
    df_sheet["lat"] = parse_coord_series(df_sheet[lat_col_guess])
    df_sheet["lon"] = parse_coord_series(df_sheet[lon_col_guess])
    df_sheet = df_sheet.dropna(subset=["lat", "lon"]).reset_index(drop=True)
else:
    st.error("Couldn't detect Latitude/Longitude columns for the selected sheet. Use a sheet that contains coordinates.")
    st.stop()

# Detect the five requested fields automatically (for current sheet)
detected = {
    "date": try_guess_col(cols, ["accident date", "date", "date of accident"]) or find_col_by_cells(df_sheet, ["accident date", "date"]),
    "time": try_guess_col(cols, ["accident time", "time", "time of accident"]) or find_col_by_cells(df_sheet, ["accident time", "time"]),
    "location": try_guess_col(cols, ["location", "place", "site", "location of accident"]) or find_col_by_cells(df_sheet, ["location", "place", "site"]),
    "accident_location": try_guess_col(cols, ["accident location", "acc_location", "accident_loc"]) or find_col_by_cells(df_sheet, ["accident location", "acc_location"]),
    "remarks": try_guess_col(cols, ["remarks", "remark", "description", "note", "notes"]) or find_col_by_cells(df_sheet, ["remarks", "description", "note"])
}

# Sidebar mapping controls to override detection (per sheet)
st.sidebar.markdown("### Detected mapping (override if needed)")
mapping = {}
for key, pretty in [("date", "Date"), ("time", "Time"), ("location", "Location"), ("accident_location", "Accident location"), ("remarks", "Remarks")]:
    options = ["<none>"] + cols
    default = detected.get(key) if detected.get(key) in cols else "<none>"
    sel = st.sidebar.selectbox(f"{pretty} column", options, index=options.index(default))
    mapping[key] = None if sel == "<none>" else sel

# Save mapping in df attrs so popup builder can use it
df_sheet.attrs["popup_cols"] = mapping

# Optional: parse date/time into helper columns
if mapping.get("date"):
    try:
        df_sheet["_date_parsed"] = pd.to_datetime(df_sheet[mapping["date"]], errors="coerce")
    except Exception:
        df_sheet["_date_parsed"] = pd.NaT
else:
    df_sheet["_date_parsed"] = pd.NaT
if mapping.get("time"):
    try:
        df_sheet["_time_parsed"] = pd.to_datetime(df_sheet[mapping["time"]].astype(str).str.replace("Hrs.", "", case=False), errors="coerce").dt.time
    except Exception:
        df_sheet["_time_parsed"] = pd.NaT
else:
    df_sheet["_time_parsed"] = pd.NaT

# Compute black spots using the local-circle rule with user threshold
clustered_df_with_flags, blackspot_summary = compute_blackspots_by_local_circles(df_sheet, radius_m=BUFFER_METERS, threshold=int(threshold))

# Map area (big, centered)
st.subheader(f"Accident Map — Sheet: {selected_sheet}")
fmap = make_map(clustered_df_with_flags, blackspot_summary)
st_folium(fmap, width=None, height=880)

st.markdown("---")
st.subheader(f"All Detected Black Spots — {selected_sheet}")
if blackspot_summary is None or blackspot_summary.empty:
    st.info(f"No black spots detected for this sheet with threshold = {int(threshold)}.")
else:
    # show all black spots table
    st.dataframe(blackspot_summary.rename(columns={
        "cand_cluster": "ClusterID",
        "accidents": "Accidents",
        "centroid_lat": "Centroid Lat",
        "centroid_lon": "Centroid Lon"
    }), use_container_width=True, hide_index=True)

st.markdown("### Notes")
st.markdown(f"- Black-spot rule: for each accident point, form a {BUFFER_METERS} m circle; if a circle contains ≥ {int(threshold)} points it is a black-spot candidate.")
st.markdown("- Candidate circles that overlap are merged into unique black spots (centroids shown as black dots).")
st.markdown("- Use the sidebar mapping controls if any popup fields show `—`.")
st.markdown(f"- Default workbook: `{DEFAULT_DATA_PATH}` (or upload a new file).")
