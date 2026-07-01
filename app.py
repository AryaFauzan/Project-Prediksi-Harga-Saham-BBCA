import streamlit as st
import pandas as pd
import joblib
import json
import yfinance as yf
import numpy as np

# ===============================
# PAGE CONFIG
# ===============================

st.set_page_config(
    page_title="BBCA Predictor",
    page_icon="📈",
    layout="wide"
)

# ===============================
# CSS
# ===============================
st.markdown("""
<style>
.stApp {
    background: #0d0d0d;
    color: #e5e5e5;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
header[data-testid="stHeader"] {
    display: none;
}
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── NAVBAR ── */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 28px;
    height: 52px;
    background: #111;
    border-bottom: 1px solid #1f1f1f;
    position: sticky;
    top: 0;
    z-index: 100;
}
.navbar-brand { font-size: 14px; font-weight: 600; color: #fff; letter-spacing: 0.02em; }

/* ── PAGE ── */
.page {
    padding: 32px 36px 0;
    max-width: 1000px;
    width: 100%;
    margin: 0 auto;
}

/* ── HERO ── */
.hero {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 36px;
    padding-bottom: 24px;
    border-bottom: 1px solid #1a1a1a;
}
.ticker-row { display: flex; align-items: center; gap: 10px; }
.ticker-symbol { font-size: 52px; font-weight: 800; color: #fff; line-height: 1; letter-spacing: -0.02em; margin: 0; }
.ticker-badge { font-size: 10px; font-weight: 600; color: #888; background: #1e1e1e; border: 1px solid #2e2e2e; padding: 3px 7px; border-radius: 4px; letter-spacing: 0.08em; align-self: flex-end; margin-bottom: 8px; }
.company-name { font-size: 12px; color: #555; margin-top: 5px; }
.hero-right { text-align: right; }
.last-price-label { font-size: 11px; color: #555; letter-spacing: 0.04em; margin-bottom: 4px; }
.last-price-value { font-size: 38px; font-weight: 700; color: #fff; letter-spacing: -0.01em; line-height: 1; }
.hero-sub { font-size: 11px; color: #555; margin-top: 6px; }
.hero-sub span { color: #777; }

/* ── SECTION ── */
.section { margin-bottom: 28px; }

.section-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 14px;
}
.section-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
}
.section-dot.prev  { background: #444; }
.section-dot.today { background: #3b82f6; }
.section-dot.tmrw  { background: #8b5cf6; }
.section-dot.model { background: #f59e0b; }

.section-title {
    font-size: 10px;
    font-weight: 700;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 0;
}

/* ── CARD GRID ── */
.grid-2 {
    display: grid;
    grid-template-columns: 3fr 2fr;
    gap: 12px;
    margin-bottom: 12px;
}
.grid-2-eq {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-bottom: 12px;
}
.grid-3 {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 12px;
    margin-bottom: 12px;
}
.grid-5 {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
}

.section > *:last-child { margin-bottom: 0; }

/* ── CARD ── */
.card {
    background: #141414;
    border: 1px solid #1e1e1e;
    border-radius: 10px;
    padding: 16px 18px;
}
.card-label {
    font-size: 10px;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
}
.card-value {
    font-size: 22px;
    font-weight: 700;
    color: #e0e0e0;
    letter-spacing: -0.01em;
    line-height: 1.1;
}
.card-value.lg { font-size: 26px; }
.card-value.sm { font-size: 18px; }
.card-value.up   { color: #4ade80; }
.card-value.down { color: #f87171; }
.card-value.muted { color: #888; }
.card-sub {
    font-size: 11px;
    color: #444;
    margin-top: 4px;
}
.card-sub.up   { color: #4ade80; }
.card-sub.down { color: #f87171; }

/* ── ERROR / NOTE ── */
.error-panel { background: #1a0a0a; border: 1px solid #3a1010; border-radius: 10px; padding: 16px; margin-bottom: 20px; font-size: 13px; color: #f87171; line-height: 1.6; }

.note-strip {
    font-size: 11px;
    color: #333;
    padding: 12px 16px;
    background: #111;
    border: 1px solid #1a1a1a;
    border-radius: 8px;
    line-height: 1.7;
    margin-top: 8px;
}
.note-strip strong { color: #444; }
</style>
""", unsafe_allow_html=True)


# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    model = joblib.load("model/bbca_random_forest_model.pkl")
    with open("model/metadata.json") as f:
        meta = json.load(f)
    return model, meta

model, meta = load_model()


# ===============================
# GET DATA
# ===============================
@st.cache_data(ttl=3600)
def get_data():
    df = yf.download("BBCA.JK", period="2y", progress=False)
    # handle multi-index columns for newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    return df

df = get_data()


# ===============================
# FEATURE
# ===============================
def create_feature(df):
    data = df.copy()

    for lag in [1, 2, 3, 5, 10, 20]:
        data[f"close_lag_{lag}"] = data["Close"].shift(lag)

    for period in [1, 3, 5, 10, 20]:
        data[f"return_{period}"] = data["Close"].pct_change(period)

    for window in [5, 10, 20, 50]:
        data[f"ma_{window}"] = data["Close"].rolling(window).mean()
        data[f"close_to_ma_{window}"] = (data["Close"] / data[f"ma_{window}"]) - 1

    for window in [5, 10, 20]:
        data[f"volatility_{window}"] = data["Close"].pct_change().rolling(window).std()

    for window in [5, 10, 20]:
        data[f"volume_ma_{window}"] = data["Volume"].rolling(window).mean()
        data[f"volume_ratio_{window}"] = data["Volume"] / data[f"volume_ma_{window}"]

    data["daily_range"] = (data["High"] - data["Low"]) / data["Close"]
    data["open_close_ratio"] = data["Open"] / data["Close"]
    data["high_close_ratio"] = data["High"] / data["Close"]
    data["low_close_ratio"] = data["Low"] / data["Close"]
    data["candle_body"] = (data["Close"] - data["Open"]) / data["Open"]

    data = data.replace([np.inf, -np.inf], np.nan)
    data.dropna(inplace=True)

    return data

X = create_feature(df)

# Predictions
# Tomorrow
latest = X.tail(1)[model.feature_names_in_]
predicted_tomorrow = float(model.predict(latest)[0])

# Today (from yesterday's perspective)
previous_row = X.iloc[-2:-1][model.feature_names_in_]
predicted_today = float(model.predict(previous_row)[0])


# ===============================
# METRICS & DATES CALCULATION
# ===============================
date_col = "Date" if "Date" in df.columns else df.columns[0]
latest_date_dt = df[date_col].iloc[-1]
previous_date_dt = df[date_col].iloc[-2]

latest_date = latest_date_dt.strftime('%d %b %Y')
previous_date = previous_date_dt.strftime('%d %b %Y')

# Next trading day assumption
next_trading_date_dt = latest_date_dt + pd.Timedelta(days=1)
if next_trading_date_dt.weekday() >= 5:
    next_trading_date_dt += pd.Timedelta(days=7 - next_trading_date_dt.weekday())
next_trading_date = next_trading_date_dt.strftime('%d %b %Y')

current_close = float(df["Close"].iloc[-1])
previous_close = float(df["Close"].iloc[-2])

# Calcs: Harga Kemarin
actual_chg = current_close - previous_close
actual_chg_pct = (actual_chg / previous_close) * 100

# Calcs: Prediksi Hari Ini
pred_today_chg = predicted_today - previous_close
pred_today_chg_pct = (pred_today_chg / previous_close) * 100
today_error = predicted_today - current_close
today_error_pct = (today_error / current_close) * 100

# Calcs: Prediksi Besok
pred_tomorrow_chg = predicted_tomorrow - current_close
pred_tomorrow_chg_pct = (pred_tomorrow_chg / current_close) * 100
tomorrow_direction = "Naik" if pred_tomorrow_chg >= 0 else "Turun"


# ===============================
# HTML RENDER HELPERS
# ===============================
def get_color(val):
    return "up" if val >= 0 else "down"

def get_sign(val):
    return "+" if val >= 0 else ""


# ===============================
# RENDER HTML
# ===============================
html_content = f"""
    <nav class="navbar">
        <span class="navbar-brand">BBCA Predictor</span>
    </nav>

    <div class="page">
        <!-- HERO -->
        <div class="hero">
            <div>
                <div class="ticker-row">
                    <span class="ticker-symbol">BBCA</span>
                    <span class="ticker-badge">IDX</span>
                </div>
                <div class="company-name">Bank Central Asia</div>
            </div>
            <div class="hero-right">
                <div class="last-price-label">Harga Terakhir ({latest_date})</div>
                <div class="last-price-value">Rp {current_close:,.2f}</div>
                <div class="hero-sub">Sumber: <span>Yahoo Finance</span></div>
            </div>
        </div>

        <!-- SECTION 1: HARGA KEMARIN -->
        <div class="section">
            <div class="section-header">
                <span class="section-dot prev"></span>
                <span class="section-title">Harga Kemarin</span>
            </div>
            <div class="grid-3">
                <div class="card">
                    <div class="card-label">Harga Close Sebelumnya</div>
                    <div class="card-value lg">Rp {previous_close:,.2f}</div>
                    <div class="card-sub">{previous_date}</div>
                </div>
                <div class="card">
                    <div class="card-label">Harga Saat Ini</div>
                    <div class="card-value lg">Rp {current_close:,.2f}</div>
                    <div class="card-sub">{latest_date}</div>
                </div>
                <div class="card">
                    <div class="card-label">Perubahan Harga</div>
                    <div class="card-value lg {get_color(actual_chg)}">
                        {get_sign(actual_chg)}Rp {actual_chg:,.2f}
                    </div>
                    <div class="card-sub {get_color(actual_chg)}">
                        {get_sign(actual_chg_pct)}{actual_chg_pct:.2f}% dari hari sebelumnya
                    </div>
                </div>
            </div>
        </div>

        <!-- SECTION 2: PREDIKSI HARI INI -->
        <div class="section">
            <div class="section-header">
                <span class="section-dot today"></span>
                <span class="section-title">Prediksi Hari Ini</span>
            </div>
            <div class="grid-2">
                <div class="card">
                    <div class="card-label">Prediksi Penutupan(Close) Hari Ini</div>
                    <div class="card-value lg">Rp {predicted_today:,.2f}</div>
                    <div class="card-sub">{latest_date}</div>
                </div>
                <div class="card">
                    <div class="card-label">Arah Prediksi</div>
                    <div class="card-value lg {get_color(pred_today_chg)}">
                        {'↑ Naik' if pred_today_chg >= 0 else '↓ Turun'}
                    </div>
                    <div class="card-sub {get_color(pred_today_chg)}">
                        {get_sign(pred_today_chg_pct)}{pred_today_chg_pct:.2f}% dari harga kemarin
                    </div>
                </div>
            </div>
            <div class="grid-3">
                <div class="card">
                    <div class="card-label">Selisih Prediksi</div>
                    <div class="card-value {get_color(pred_today_chg)}">
                        {get_sign(pred_today_chg)}Rp {pred_today_chg:,.2f}
                    </div>
                    <div class="card-sub">Prediksi hari ini − Harga kemarin</div>
                </div>
                <div class="card">
                    <div class="card-label">Tingkat Kesalahan</div>
                    <div class="card-value muted">{abs(today_error_pct):.2f}%</div>
                    <div class="card-sub">MAPE hari ini</div>
                </div>
            </div>
        </div>

        <!-- SECTION 3: PREDIKSI BESOK -->
        <div class="section">
            <div class="section-header">
                <span class="section-dot tmrw"></span>
                <span class="section-title">Prediksi Besok</span>
            </div>
            <div class="grid-2">
                <div class="card">
                    <div class="card-label">Prediksi Penutupan(Close) Besok</div>
                    <div class="card-value lg">Rp {predicted_tomorrow:,.2f}</div>
                    <div class="card-sub">{next_trading_date}</div>
                </div>
                <div class="card">
                    <div class="card-label">Arah</div>
                    <div class="card-value lg {get_color(pred_tomorrow_chg)}">
                        {'↑' if pred_tomorrow_chg >= 0 else '↓'} {tomorrow_direction}
                    </div>
                    <div class="card-sub {get_color(pred_tomorrow_chg)}">
                        {get_sign(pred_tomorrow_chg_pct)}{pred_tomorrow_chg_pct:.2f}% dari harga terakhir
                    </div>
                </div>
            </div>
            <div class="grid-3">
                <div class="card">
                    <div class="card-label">Selisih Prediksi</div>
                    <div class="card-value {get_color(pred_tomorrow_chg)}">
                        {get_sign(pred_tomorrow_chg)}Rp {pred_tomorrow_chg:,.2f}
                    </div>
                    <div class="card-sub">Prediksi Penutupan Besok − Harga saat ini</div>
                </div>
            </div>
        </div>

        <!-- INFORMASI MODEL -->
        <div class="section">
            <div class="section-header">
                <span class="section-dot model"></span>
                <span class="section-title">Informasi Model</span>
            </div>
            <div class="grid-5">
                <div class="card">
                    <div class="card-label">Model</div>
                    <div class="card-value sm muted">Random Forest</div>
                </div>
                <div class="card">
                    <div class="card-label">Skor R²</div>
                    <div class="card-value sm muted">0.9504</div>
                </div>
                <div class="card">
                    <div class="card-label">MAE</div>
                    <div class="card-value sm muted">Rp 177</div>
                </div>
                <div class="card">
                    <div class="card-label">RMSE</div>
                    <div class="card-value sm muted">Rp 247</div>
                </div>
                <div class="card">
                    <div class="card-label">MAPE Model</div>
                    <div class="card-value sm muted">2.11%</div>
                </div>
            </div>
        </div>

        <div class="note-strip">
            <strong>Catatan:</strong> Hasil prediksi ini digunakan untuk tujuan akademik dan demonstrasi deployment model machine learning.
            Bukan rekomendasi investasi. Harga saham dipengaruhi kondisi pasar, laporan keuangan, dan faktor eksternal lainnya.
        </div>

    </div>
"""

html_content = "\n".join(line.strip() for line in html_content.split("\n"))

st.markdown(html_content, unsafe_allow_html=True)