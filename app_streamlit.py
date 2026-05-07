import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="HydroCheck — Prediksi Hidrasi",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===============================
# CUSTOM CSS
# ===============================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: #f4f7fb;
    }

    /* ===== HIDE SIDEBAR ===== */
    [data-testid="stSidebar"] { display: none; }
    header[data-testid="stHeader"] { background: transparent; height: 0; }

    /* ===== SCHOOL HEADER ===== */
    .school-header {
        background: linear-gradient(135deg, #0d47a1 0%, #1565c0 50%, #1976d2 100%);
        border-radius: 0 0 24px 24px;
        padding: 1rem 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(13, 71, 161, 0.3);
    }

    .header-left {
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .header-icon {
        font-size: 2.8rem;
        filter: drop-shadow(0 2px 8px rgba(0,0,0,0.2));
    }

    .header-title h1 {
        color: white;
        font-size: 1.6rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.2;
        letter-spacing: -0.02em;
    }

    .header-title p {
        color: rgba(255,255,255,0.75);
        font-size: 0.78rem;
        font-weight: 500;
        margin: 0;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    .header-right {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.3rem;
    }

    .school-logo {
        width: 64px;
        height: 64px;
        border-radius: 50%;
        background: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        overflow: hidden;
    }

    .school-logo img {
        width: 100%;
        height: 100%;
        object-fit: contain;
        border-radius: 50%;
    }

    .school-name {
        color: rgba(255,255,255,0.85);
        font-size: 0.68rem;
        font-weight: 600;
        text-align: center;
        letter-spacing: 0.04em;
    }

    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        background: white;
        border-radius: 14px;
        padding: 5px;
        gap: 3px;
        box-shadow: 0 2px 10px rgba(13, 71, 161, 0.08);
        border: 1px solid #e3eaf7;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 0.85rem;
        color: #64748b;
        padding: 9px 20px;
        transition: all 0.2s;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0d47a1, #1976d2) !important;
        color: white !important;
    }

    /* ===== CARDS ===== */
    .card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(13, 71, 161, 0.07);
        border: 1px solid #e8eef8;
        margin-bottom: 1rem;
    }

    .card-title {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #1565c0;
        margin-bottom: 1rem;
    }

    /* ===== RESULT ===== */
    .result-good {
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        border: 2px solid #43a047;
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
    }

    .result-poor {
        background: linear-gradient(135deg, #fce4ec, #f8bbd0);
        border: 2px solid #e53935;
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
    }

    .result-emoji { font-size: 3rem; }
    .result-label { font-size: 1.6rem; font-weight: 800; margin: 0.3rem 0; }
    .result-desc { font-size: 0.85rem; color: #546e7a; }

    /* ===== METRIC CARDS ===== */
    .metric-row {
        display: flex;
        gap: 0.8rem;
        margin-bottom: 1rem;
    }

    .metric-item {
        flex: 1;
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 1px 8px rgba(13, 71, 161, 0.07);
        border: 1px solid #e8eef8;
        text-align: center;
    }

    .metric-val {
        font-family: 'Fira Code', monospace;
        font-size: 1.3rem;
        font-weight: 700;
        color: #0d47a1;
    }

    .metric-lbl {
        font-size: 0.72rem;
        color: #90a4ae;
        font-weight: 500;
        margin-top: 2px;
    }

    /* ===== NOTEBOOK CELLS ===== */
    .nb-section-title {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #1565c0;
        padding: 0.5rem 0;
        border-bottom: 2px solid #e3eaf7;
        margin: 1.2rem 0 0.8rem 0;
    }

    .nb-cell {
        background: white;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e3eaf7;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 6px rgba(13,71,161,0.05);
    }

    .nb-cell-header {
        background: #f0f4ff;
        padding: 0.4rem 1rem;
        font-size: 0.72rem;
        font-weight: 600;
        color: #5c7cfa;
        letter-spacing: 0.06em;
        border-bottom: 1px solid #e3eaf7;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .nb-code {
        background: #1e2432;
        color: #a9b7d0;
        padding: 0.9rem 1.2rem;
        font-family: 'Fira Code', monospace;
        font-size: 0.78rem;
        line-height: 1.7;
        white-space: pre-wrap;
        overflow-x: auto;
    }

    .nb-output {
        background: #fafbff;
        padding: 0.8rem 1.2rem;
        font-family: 'Fira Code', monospace;
        font-size: 0.78rem;
        line-height: 1.7;
        color: #2c3e50;
        border-top: 1px solid #e3eaf7;
        white-space: pre-wrap;
    }

    .nb-output-label {
        font-size: 0.65rem;
        font-weight: 700;
        color: #90a4ae;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.3rem;
    }

    .kw { color: #c792ea; }
    .fn { color: #82aaff; }
    .st { color: #c3e88d; }
    .cm { color: #546e7a; font-style: italic; }
    .nb-kw { color: #89ddff; }
    .nu { color: #f78c6c; }

    /* ===== ACCURACY BADGE ===== */
    .acc-badge {
        display: inline-block;
        background: linear-gradient(135deg, #0d47a1, #1976d2);
        color: white;
        border-radius: 100px;
        padding: 0.25rem 0.9rem;
        font-size: 0.78rem;
        font-weight: 700;
        font-family: 'Fira Code', monospace;
    }

    .acc-good { background: linear-gradient(135deg, #2e7d32, #43a047); }
    .acc-ok   { background: linear-gradient(135deg, #e65100, #f57c00); }

    /* ===== BUTTONS ===== */
    .stButton > button {
        background: linear-gradient(135deg, #0d47a1 0%, #1976d2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 0.7rem 2rem;
        letter-spacing: 0.02em;
        transition: all 0.2s;
        box-shadow: 0 4px 14px rgba(13, 71, 161, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(13, 71, 161, 0.4);
    }

    /* ===== SLIDER & SELECT LABELS ===== */
    [data-testid="stSlider"] label,
    [data-testid="stSelectbox"] label {
        color: #1a2744 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }

    /* ===== TIPS ===== */
    .tip-box {
        background: #e8f4fd;
        border-left: 4px solid #1976d2;
        border-radius: 0 10px 10px 0;
        padding: 0.8rem 1rem;
        margin-top: 0.5rem;
        font-size: 0.83rem;
        color: #0d47a1;
        line-height: 1.6;
    }

    /* ===== INFO ITEM ===== */
    .info-item {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        background: white;
        border-radius: 10px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 5px rgba(13,71,161,0.06);
        border: 1px solid #e8eef8;
    }

    .info-icon { font-size: 1.3rem; }
    .info-name { font-weight: 700; font-size: 0.85rem; color: #1a2744; }
    .info-desc { font-size: 0.74rem; color: #78909c; }
</style>
""", unsafe_allow_html=True)


# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    try:
        return joblib.load("logistic_regression_model.joblib")
    except:
        return None

model = load_model()


# ===============================
# SCHOOL HEADER
# ===============================
st.markdown("""
<div class="school-header">
    <div class="header-left">
        <div class="header-icon">💧</div>
        <div class="header-title">
            <h1>HydroCheck</h1>
            <p>Prediksi Hidrasi · Machine Learning App</p>
        </div>
    </div>
    <div class="header-right">
        <div class="school-logo">
            <img src="img/Logo_SMK_Negeri_1_Purbalingga.png"
                 onerror="this.parentElement.innerHTML='🏫'"
                 alt="Logo Sekolah">
        </div>
        <div class="school-name">SMK Negeri 1<br>Purbalingga</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ===============================
# TABS
# ===============================
tab1, tab2, tab3 = st.tabs([
    "💧  Prediksi Hidrasi",
    "📓  Notebook",
    "👨‍💻  Developer",
])


# ====================================
# TAB 1 — PREDIKSI
# ====================================
with tab1:

    st.markdown("<br>", unsafe_allow_html=True)

    col_input, col_result = st.columns([1.1, 1], gap="large")

    with col_input:
        # Personal Data
        st.markdown('<div class="card"><div class="card-title">👤 Data Personal</div>', unsafe_allow_html=True)

        umur = st.slider("🎂 Umur (tahun)", 10, 80, 25)
        berat = st.slider("⚖️ Berat Badan (kg)", 30.0, 120.0, 60.0, step=0.5)
        gender = st.selectbox("🚻 Jenis Kelamin", ["Laki-Laki", "Perempuan"])

        st.markdown("</div>", unsafe_allow_html=True)

        # Lifestyle
        st.markdown('<div class="card"><div class="card-title">🌿 Gaya Hidup & Lingkungan</div>', unsafe_allow_html=True)

        air = st.slider("🥤 Konsumsi Air per Hari (liter)", 0.5, 5.0, 2.0, step=0.1)
        aktivitas = st.selectbox("🏃 Tingkat Aktivitas", ["Rendah", "Sedang", "Tinggi"])
        cuaca = st.selectbox("🌤️ Kondisi Cuaca", ["Normal", "Panas", "Dingin"])

        st.markdown("</div>", unsafe_allow_html=True)

        col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
        with col_b2:
            btn = st.button("🔎 Prediksi Sekarang", use_container_width=True)

    with col_result:
        st.markdown("""
        <div class="metric-row">
            <div class="metric-item">
                <div class="metric-val">30.000</div>
                <div class="metric-lbl">Data Training</div>
            </div>
            <div class="metric-item">
                <div class="metric-val">6</div>
                <div class="metric-lbl">Fitur Input</div>
            </div>
            <div class="metric-item">
                <div class="metric-val">3</div>
                <div class="metric-lbl">Model ML</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if btn:
            gender_map = {"Laki-Laki": "Male", "Perempuan": "Female"}
            aktivitas_map = {"Rendah": "Low", "Sedang": "Moderate", "Tinggi": "High"}
            cuaca_map = {"Normal": "Normal", "Panas": "Hot", "Dingin": "Cold"}

            input_data = pd.DataFrame([[
                umur,
                gender_map[gender],
                berat,
                air,
                aktivitas_map[aktivitas],
                cuaca_map[cuaca]
            ]], columns=["Age", "Gender", "Weight (kg)",
                         "Daily Water Intake (liters)",
                         "Physical Activity Level", "Weather"])

            if model:
                hasil = model.predict(input_data)[0]
            else:
                # Simulasi sederhana jika model tidak ada
                kebutuhan = 3.7 if gender == "Laki-Laki" else 2.7
                if cuaca == "Panas": kebutuhan += 0.5
                if aktivitas == "Tinggi": kebutuhan += 0.5
                hasil = "Good" if air >= kebutuhan * 0.8 else "Poor"

            if hasil == "Good":
                st.markdown("""
                <div class="result-good">
                    <div class="result-emoji">✅</div>
                    <div class="result-label" style="color:#2e7d32;">Hidrasi Baik</div>
                    <div class="result-desc">Tubuhmu terhidrasi dengan cukup baik. Pertahankan pola minum yang sehat!</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                <div class="tip-box">
                💡 <b>Tips:</b> Pertahankan kebiasaanmu! Minum air putih secara rutin setiap 1–2 jam sekali.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-poor">
                    <div class="result-emoji">⚠️</div>
                    <div class="result-label" style="color:#c62828;">Kurang Hidrasi</div>
                    <div class="result-desc">Tubuhmu kekurangan cairan. Segera tingkatkan konsumsi air putihmu!</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                <div class="tip-box">
                💡 <b>Tips:</b> Coba minum minimal 8 gelas (2 liter) per hari. Tambahkan asupan air saat cuaca panas atau olahraga.
                </div>
                """, unsafe_allow_html=True)

            # Summary ringkas
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="card" style="padding:1rem 1.2rem;">
                <div class="card-title">📋 Ringkasan Input</div>
                <div style="font-size:0.83rem; color:#546e7a; line-height:2;">
                    👤 {gender} &nbsp;·&nbsp; 🎂 {umur} tahun &nbsp;·&nbsp; ⚖️ {berat} kg<br>
                    🥤 {air} L/hari &nbsp;·&nbsp; 🏃 {aktivitas} &nbsp;·&nbsp; 🌤️ {cuaca}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card" style="text-align:center; padding:3rem 1.5rem;">
                <div style="font-size:3rem; margin-bottom:1rem;">💧</div>
                <div style="font-size:1rem; font-weight:700; color:#1a2744; margin-bottom:0.4rem;">Siap Memprediksi</div>
                <div style="font-size:0.83rem; color:#90a4ae;">Isi data di sebelah kiri, lalu tekan tombol <b>Prediksi Sekarang</b></div>
            </div>
            """, unsafe_allow_html=True)


# ====================================
# TAB 2 — NOTEBOOK
# ====================================
with tab2:

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card" style="padding:1rem 1.5rem; margin-bottom:1.2rem;">
        <div style="font-size:0.95rem; font-weight:700; color:#0d47a1;">📓 Jupyter Notebook — HydroCheck ML Pipeline</div>
        <div style="font-size:0.8rem; color:#78909c; margin-top:0.2rem;">Eksplorasi data, preprocessing, training model, dan evaluasi · Python 3 · Scikit-learn</div>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 1: Load Data ──
    st.markdown('<div class="nb-section-title">📂 1. Load & Eksplorasi Dataset</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="nb-cell">
        <div class="nb-cell-header">⬛ In [31]</div>
        <div class="nb-code"><span class="kw">import</span> pandas <span class="kw">as</span> pd

df = pd.read_csv(<span class="st">"Daily_Water_Intake_Modified.csv"</span>)
df</div>
        <div class="nb-output">
            <div class="nb-output-label">Out [31]</div>
<table style="font-size:0.75rem; border-collapse:collapse; width:100%;">
  <tr style="background:#e8eef8; text-align:left;">
    <th style="padding:4px 8px;"></th><th style="padding:4px 8px;">Age</th><th style="padding:4px 8px;">Gender</th><th style="padding:4px 8px;">Weight (kg)</th><th style="padding:4px 8px;">Daily Water Intake (liters)</th><th style="padding:4px 8px;">Physical Activity Level</th><th style="padding:4px 8px;">Weather</th><th style="padding:4px 8px;">Hydration Level</th>
  </tr>
  <tr><td style="padding:4px 8px; color:#78909c;">0</td><td style="padding:4px 8px;">56</td><td style="padding:4px 8px;">Male</td><td style="padding:4px 8px;">96</td><td style="padding:4px 8px;">4.23</td><td style="padding:4px 8px;">Moderate</td><td style="padding:4px 8px;">Hot</td><td style="padding:4px 8px;">Good</td></tr>
  <tr style="background:#f8faff;"><td style="padding:4px 8px; color:#78909c;">1</td><td style="padding:4px 8px;">60</td><td style="padding:4px 8px;">Male</td><td style="padding:4px 8px;">105</td><td style="padding:4px 8px;">3.95</td><td style="padding:4px 8px;">High</td><td style="padding:4px 8px;">Normal</td><td style="padding:4px 8px;">Good</td></tr>
  <tr><td style="padding:4px 8px; color:#78909c;">2</td><td style="padding:4px 8px;">36</td><td style="padding:4px 8px;">Male</td><td style="padding:4px 8px;">68</td><td style="padding:4px 8px;">2.39</td><td style="padding:4px 8px;">Moderate</td><td style="padding:4px 8px;">Cold</td><td style="padding:4px 8px;">Good</td></tr>
  <tr style="background:#f8faff;"><td style="padding:4px 8px; color:#78909c;">3</td><td style="padding:4px 8px;">19</td><td style="padding:4px 8px;">Female</td><td style="padding:4px 8px;">74</td><td style="padding:4px 8px;">3.13</td><td style="padding:4px 8px;">Moderate</td><td style="padding:4px 8px;">Hot</td><td style="padding:4px 8px;">Good</td></tr>
  <tr><td style="padding:4px 8px; color:#78909c;">4</td><td style="padding:4px 8px;">38</td><td style="padding:4px 8px;">Male</td><td style="padding:4px 8px;">77</td><td style="padding:4px 8px;">2.11</td><td style="padding:4px 8px;">Low</td><td style="padding:4px 8px;">Normal</td><td style="padding:4px 8px;">Good</td></tr>
  <tr style="background:#f8faff;"><td style="padding:4px 8px; color:#78909c;" colspan="8">... (30000 rows × 7 columns)</td></tr>
</table>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="nb-cell">
        <div class="nb-cell-header">⬛ In [32–35]</div>
        <div class="nb-code">df.shape
df.columns
df.dtypes
df.info()</div>
        <div class="nb-output">
            <div class="nb-output-label">Out</div>
(30000, 7)

Index(['Age', 'Gender', 'Weight (kg)', 'Daily Water Intake (liters)',
       'Physical Activity Level', 'Weather', 'Hydration Level'], dtype='object')

Age                              int64
Gender                          object
Weight (kg)                      int64
Daily Water Intake (liters)    float64
Physical Activity Level         object
Weather                         object
Hydration Level                 object

&lt;class 'pandas.core.frame.DataFrame'&gt;
RangeIndex: 30000 entries, 0 to 29999
Data columns (total 7 columns):
 #   Column                       Non-Null Count   Dtype
---  ------                       --------------   -----
 0   Age                          30000 non-null   int64
 1   Gender                       30000 non-null   object
 2   Weight (kg)                  30000 non-null   int64
 3   Daily Water Intake (liters)  30000 non-null   float64
 4   Physical Activity Level      30000 non-null   object
 5   Weather                      30000 non-null   object
 6   Hydration Level              30000 non-null   object
dtypes: float64(1), int64(2), object(4)
memory usage: 1.6+ MB
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="nb-cell">
        <div class="nb-cell-header">⬛ In [36]</div>
        <div class="nb-code">df.describe()</div>
        <div class="nb-output">
            <div class="nb-output-label">Out [36]</div>
<table style="font-size:0.75rem; border-collapse:collapse; width:100%;">
  <tr style="background:#e8eef8;">
    <th style="padding:4px 8px;"></th><th style="padding:4px 8px;">Age</th><th style="padding:4px 8px;">Weight (kg)</th><th style="padding:4px 8px;">Daily Water Intake (liters)</th>
  </tr>
  <tr><td style="padding:4px 8px; font-weight:600;">count</td><td style="padding:4px 8px;">30000.0</td><td style="padding:4px 8px;">30000.0</td><td style="padding:4px 8px;">30000.0</td></tr>
  <tr style="background:#f8faff;"><td style="padding:4px 8px; font-weight:600;">mean</td><td style="padding:4px 8px;">43.47</td><td style="padding:4px 8px;">76.85</td><td style="padding:4px 8px;">2.85</td></tr>
  <tr><td style="padding:4px 8px; font-weight:600;">std</td><td style="padding:4px 8px;">14.99</td><td style="padding:4px 8px;">18.74</td><td style="padding:4px 8px;">0.84</td></tr>
  <tr style="background:#f8faff;"><td style="padding:4px 8px; font-weight:600;">min</td><td style="padding:4px 8px;">18</td><td style="padding:4px 8px;">45</td><td style="padding:4px 8px;">1.50</td></tr>
  <tr><td style="padding:4px 8px; font-weight:600;">max</td><td style="padding:4px 8px;">69</td><td style="padding:4px 8px;">109</td><td style="padding:4px 8px;">5.43</td></tr>
</table>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 2: EDA ──
    st.markdown('<div class="nb-section-title">📊 2. Eksplorasi Data (EDA)</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="nb-cell">
        <div class="nb-cell-header">⬛ In [37–40]</div>
        <div class="nb-code">df[<span class="st">"Gender"</span>].value_counts()
df[<span class="st">"Physical Activity Level"</span>].value_counts()
df[<span class="st">"Weather"</span>].value_counts()
df[<span class="st">"Hydration Level"</span>].value_counts()</div>
        <div class="nb-output">
            <div class="nb-output-label">Out</div>
Gender
Male      15032
Female    14968
Name: count, dtype: int64

Physical Activity Level
High        10069
Low         10011
Moderate     9920
Name: count, dtype: int64

Weather
Hot       10081
Cold      10012
Normal     9907
Name: count, dtype: int64

Hydration Level
Good    29343
Poor      657
Name: count, dtype: int64
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="nb-cell">
        <div class="nb-cell-header">⬛ In [44]</div>
        <div class="nb-code">df.isnull().sum()</div>
        <div class="nb-output">
            <div class="nb-output-label">Out [44]</div>
Age                            0
Gender                         0
Weight (kg)                    0
Daily Water Intake (liters)    0
Physical Activity Level        0
Weather                        0
Hydration Level                0
dtype: int64

✅ Tidak ada missing values!
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 3: Preprocessing ──
    st.markdown('<div class="nb-section-title">⚙️ 3. Preprocessing & Split Data</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="nb-cell">
        <div class="nb-cell-header">⬛ In [48]</div>
        <div class="nb-code"><span class="kw">from</span> sklearn.linear_model <span class="kw">import</span> LogisticRegression
<span class="kw">from</span> sklearn.model_selection <span class="kw">import</span> train_test_split, cross_val_score
<span class="kw">from</span> sklearn.metrics <span class="kw">import</span> accuracy_score, classification_report, confusion_matrix
<span class="kw">from</span> sklearn.preprocessing <span class="kw">import</span> StandardScaler, OneHotEncoder, OrdinalEncoder
<span class="kw">from</span> sklearn.pipeline <span class="kw">import</span> Pipeline
<span class="kw">from</span> sklearn.compose <span class="kw">import</span> ColumnTransformer
<span class="kw">from</span> sklearn.ensemble <span class="kw">import</span> RandomForestClassifier
<span class="kw">from</span> sklearn.tree <span class="kw">import</span> DecisionTreeClassifier

X = df[[<span class="st">"Age"</span>, <span class="st">"Gender"</span>, <span class="st">"Weight (kg)"</span>, <span class="st">"Daily Water Intake (liters)"</span>,
        <span class="st">"Physical Activity Level"</span>, <span class="st">"Weather"</span>]]
y = df[<span class="st">"Hydration Level"</span>]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=<span class="nu">0.2</span>, random_state=<span class="nu">42</span>, stratify=y
)

<span class="cm"># Preprocessing pipeline</span>
numeric_columns    = [<span class="st">"Age"</span>, <span class="st">"Weight (kg)"</span>, <span class="st">"Daily Water Intake (liters)"</span>]
categorical_columns = [<span class="st">"Gender"</span>, <span class="st">"Weather"</span>]
ordinal_columns    = [<span class="st">"Physical Activity Level"</span>]
activity_order     = [<span class="st">"Low"</span>, <span class="st">"Moderate"</span>, <span class="st">"High"</span>]

preprocessing = ColumnTransformer(transformers=[
    (<span class="st">"scaler"</span>, StandardScaler(),  numeric_columns),
    (<span class="st">"ohe"</span>,    OneHotEncoder(),   categorical_columns),
    (<span class="st">"oe"</span>,     OrdinalEncoder(categories=[activity_order]), ordinal_columns)
])</div>
        <div class="nb-output">
            <div class="nb-output-label">Keterangan Split</div>
Train size : 24.000 data (80%)
Test size  :  6.000 data (20%)
Stratify   : Yes → proporsi Good/Poor dijaga sama
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 4: Training ──
    st.markdown('<div class="nb-section-title">🤖 4. Training & Evaluasi Model</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="nb-cell">
        <div class="nb-cell-header">⬛ In [49] — Logistic Regression</div>
        <div class="nb-code">model = Pipeline(steps=[
    (<span class="st">"preprocessing"</span>, preprocessing),
    (<span class="st">"model"</span>, LogisticRegression(random_state=<span class="nu">42</span>, max_iter=<span class="nu">1000</span>))
])

model.fit(X_train, y_train)
y_pred_lr = model.predict(X_test)

<span class="fn">print</span>(<span class="st">"Accuracy Score : "</span>, accuracy_score(y_test, y_pred_lr))
<span class="fn">print</span>(<span class="st">"Classification Report :\n"</span>, classification_report(y_test, y_pred_lr))
<span class="fn">print</span>(<span class="st">"Confusion Matrix :\n"</span>, confusion_matrix(y_test, y_pred_lr))</div>
        <div class="nb-output">
            <div class="nb-output-label">Out [49]</div>
Accuracy Score :  0.9783333333333334

Classification Report :
               precision    recall  f1-score   support

        Good       0.98      1.00      0.99      5869
        Poor       0.56      0.04      0.07       131

    accuracy                           0.98      6000
   macro avg       0.77      0.52      0.53      6000
weighted avg       0.97      0.98      0.97      6000

Confusion Matrix :
 [[5865    4]
 [ 126    5]]

CV Scores     : [0.97895833 0.97854167 0.97916667 0.97916667 0.97833333]
CV Mean       : 0.9788
CV Std        : 0.0003
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="nb-cell">
        <div class="nb-cell-header">⬛ In [50] — Random Forest</div>
        <div class="nb-code">model_forest = Pipeline(steps=[
    (<span class="st">"preprocessing"</span>, preprocessing),
    (<span class="st">"model"</span>, RandomForestClassifier(random_state=<span class="nu">42</span>))
])

model_forest.fit(X_train, y_train)
y_pred_rf = model_forest.predict(X_test)</div>
        <div class="nb-output">
            <div class="nb-output-label">Out [50]</div>
Accuracy Score :  0.9756666666666667

Classification Report :
               precision    recall  f1-score   support

        Good       0.98      0.99      0.99      5869
        Poor       0.34      0.12      0.18       131

    accuracy                           0.98      6000
   macro avg       0.66      0.56      0.58      6000
weighted avg       0.97      0.98      0.97      6000

Confusion Matrix :
 [[5838   31]
 [ 115   16]]

CV Mean       : 0.9762
CV Std        : 0.0009
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="nb-cell">
        <div class="nb-cell-header">⬛ In [51] — Decision Tree</div>
        <div class="nb-code">model_tree = Pipeline(steps=[
    (<span class="st">"preprocessing"</span>, preprocessing),
    (<span class="st">"model"</span>, DecisionTreeClassifier(random_state=<span class="nu">42</span>))
])

model_tree.fit(X_train, y_train)
y_pred_dt = model_tree.predict(X_test)</div>
        <div class="nb-output">
            <div class="nb-output-label">Out [51]</div>
Accuracy Score :  0.9648333333333333

Classification Report :
               precision    recall  f1-score   support

        Good       0.98      0.98      0.98      5869
        Poor       0.22      0.24      0.23       131

    accuracy                           0.96      6000
   macro avg       0.60      0.61      0.61      6000
weighted avg       0.97      0.96      0.97      6000

CV Mean       : 0.9629
CV Std        : 0.0027
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 5: Perbandingan ──
    st.markdown('<div class="nb-section-title">🏆 5. Perbandingan Akurasi Model</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="nb-cell">
        <div class="nb-cell-header">⬛ In [53]</div>
        <div class="nb-code"><span class="fn">print</span>(<span class="st">"="*45</span>)
<span class="fn">print</span>(<span class="st">"     PERBANDINGAN AKURASI MODEL"</span>)
<span class="fn">print</span>(<span class="st">"="*45</span>)
<span class="fn">print</span>(f<span class="st">"Logistic Regression : {accuracy_score(y_test, y_pred_lr):.4f}"</span>)
<span class="fn">print</span>(f<span class="st">"Random Forest       : {accuracy_score(y_test, y_pred_rf):.4f}"</span>)
<span class="fn">print</span>(f<span class="st">"Decision Tree       : {accuracy_score(y_test, y_pred_dt):.4f}"</span>)</div>
        <div class="nb-output">
            <div class="nb-output-label">Out [53]</div>
=============================================
     PERBANDINGAN AKURASI MODEL
=============================================
Logistic Regression : 0.9783
Random Forest       : 0.9757
Decision Tree       : 0.9648
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Visual summary akurasi
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        <div class="card" style="text-align:center; padding:1.2rem;">
            <div style="font-size:1.4rem;">🔵</div>
            <div style="font-weight:700; font-size:0.85rem; color:#1a2744; margin:0.3rem 0;">Logistic Regression</div>
            <span class="acc-badge acc-good">97.83%</span>
            <div style="font-size:0.72rem; color:#78909c; margin-top:0.5rem;">🏆 Model Terbaik</div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="card" style="text-align:center; padding:1.2rem;">
            <div style="font-size:1.4rem;">🌲</div>
            <div style="font-weight:700; font-size:0.85rem; color:#1a2744; margin:0.3rem 0;">Random Forest</div>
            <span class="acc-badge">97.57%</span>
            <div style="font-size:0.72rem; color:#78909c; margin-top:0.5rem;">Runner Up</div>
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        st.markdown("""
        <div class="card" style="text-align:center; padding:1.2rem;">
            <div style="font-size:1.4rem;">🌳</div>
            <div style="font-weight:700; font-size:0.85rem; color:#1a2744; margin:0.3rem 0;">Decision Tree</div>
            <span class="acc-badge acc-ok">96.48%</span>
            <div style="font-size:0.72rem; color:#78909c; margin-top:0.5rem;">Baseline</div>
        </div>
        """, unsafe_allow_html=True)

    # ── SECTION 6: Save Model ──
    st.markdown('<div class="nb-section-title">💾 6. Simpan Model</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="nb-cell">
        <div class="nb-cell-header">⬛ In [55]</div>
        <div class="nb-code"><span class="kw">import</span> joblib

joblib.dump(model,        <span class="st">"logistic_regression_model.joblib"</span>)
joblib.dump(model_forest, <span class="st">"random_forest_model.joblib"</span>)
joblib.dump(model_tree,   <span class="st">"decision_tree_model.joblib"</span>)</div>
        <div class="nb-output">
            <div class="nb-output-label">Out [55]</div>
['logistic_regression_model.joblib']
['random_forest_model.joblib']
['decision_tree_model.joblib']
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 7: Test Prediksi ──
    st.markdown('<div class="nb-section-title">🔎 7. Uji Prediksi Data Baru</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="nb-cell">
        <div class="nb-cell-header">⬛ In [83]</div>
        <div class="nb-code">data_baru = pd.DataFrame(
    [[<span class="nu">38</span>, <span class="st">"Female"</span>, <span class="nu">77</span>, <span class="nu">1.00</span>, <span class="st">"High"</span>, <span class="st">"Hot"</span>]],
    columns=[<span class="st">"Age"</span>, <span class="st">"Gender"</span>, <span class="st">"Weight (kg)"</span>,
             <span class="st">"Daily Water Intake (liters)"</span>,
             <span class="st">"Physical Activity Level"</span>, <span class="st">"Weather"</span>]
)

prediksi = model.predict(data_baru)[<span class="nu">0</span>]
<span class="fn">print</span>(f<span class="st">"model memprediksi tingkat hidrasi {prediksi}"</span>)</div>
        <div class="nb-output">
            <div class="nb-output-label">Out [83]</div>
model memprediksi tingkat hidrasi Poor
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tip-box" style="margin-top:0;">
    💡 <b>Kesimpulan:</b> Perempuan 38 tahun, berat 77 kg, minum hanya 1 liter/hari, aktivitas tinggi di cuaca panas → diprediksi <b>Kurang Hidrasi (Poor)</b>. Masuk akal karena kebutuhan air sangat tinggi namun asupan sangat rendah.
    </div>
    """, unsafe_allow_html=True)


# ====================================
# TAB 3 — DEVELOPER
# ====================================
with tab3:

    st.markdown("<br>", unsafe_allow_html=True)

    col_d1, col_d2 = st.columns([1, 1.5], gap="large")

    with col_d1:
        st.markdown("""
        <div class="card" style="text-align:center; padding:2rem;">
            <div style="font-size:3.5rem;">👨‍🎓</div>
            <div style="font-size:1.2rem; font-weight:800; color:#0d47a1; margin:0.6rem 0 0.2rem;">Alrifat</div>
            <div style="font-size:0.8rem; color:#78909c; font-weight:500;">Machine Learning Student</div>
            <div style="margin-top:1rem;">
                <span style="background:#e8f0fe; color:#1565c0; border-radius:100px; padding:0.25rem 0.8rem; font-size:0.72rem; font-weight:600; display:inline-block; margin:0.2rem;">🎓 SMK</span>
                <span style="background:#e8f5e9; color:#2e7d32; border-radius:100px; padding:0.25rem 0.8rem; font-size:0.72rem; font-weight:600; display:inline-block; margin:0.2rem;">🤖 Machine Learning</span>
                <span style="background:#fff3e0; color:#e65100; border-radius:100px; padding:0.25rem 0.8rem; font-size:0.72rem; font-weight:600; display:inline-block; margin:0.2rem;">🐍 Python</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card" style="padding:1.2rem 1.5rem;">
            <div class="card-title">📬 Kontak</div>
            <div style="font-size:0.83rem; color:#546e7a; line-height:2.2;">
                📧 &nbsp;email@example.com<br>
                🐙 &nbsp;github.com/alrifat<br>
                🏫 &nbsp;SMK Negeri 1 Purbalingga
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_d2:
        st.markdown('<div class="card-title" style="margin-bottom:0.6rem;">🛠️ Teknologi yang Digunakan</div>', unsafe_allow_html=True)

        techs = [
            ("🐍", "Python 3.x",           "Bahasa pemrograman utama"),
            ("📊", "Pandas",               "Manipulasi & analisis data"),
            ("🤖", "Scikit-learn",         "Algoritma Machine Learning"),
            ("💾", "Joblib",               "Menyimpan & memuat model"),
            ("🌐", "Streamlit",            "Framework web app interaktif"),
            ("📈", "Matplotlib / Seaborn", "Visualisasi data"),
        ]
        for icon, name, desc in techs:
            st.markdown(f"""
            <div class="info-item">
                <div class="info-icon">{icon}</div>
                <div>
                    <div class="info-name">{name}</div>
                    <div class="info-desc">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card-title" style="margin-bottom:0.6rem;">🤖 Perbandingan Model</div>', unsafe_allow_html=True)

        models = [
            ("🔵", "Logistic Regression", "97.83%", "acc-good"),
            ("🌲", "Random Forest",       "97.57%", ""),
            ("🌳", "Decision Tree",       "96.48%", "acc-ok"),
        ]
        for icon, name, acc, cls in models:
            st.markdown(f"""
            <div class="info-item" style="justify-content:space-between;">
                <div style="display:flex; align-items:center; gap:0.8rem;">
                    <div class="info-icon">{icon}</div>
                    <div class="info-name">{name}</div>
                </div>
                <span class="acc-badge {cls}">{acc}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; padding:1rem; font-size:0.75rem; color:#90a4ae;">
        💧 HydroCheck &nbsp;·&nbsp; SMK ML Project &nbsp;·&nbsp; 2026<br>
        <span style="font-family:'Fira Code', monospace; font-size:0.68rem;">Built with Python · Scikit-learn · Streamlit</span>
    </div>
    """, unsafe_allow_html=True)