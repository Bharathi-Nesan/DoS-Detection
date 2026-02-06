import streamlit as st
import pandas as pd
import joblib
import time
import random

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="WSN-IDS Multi-Model Sentry",
    page_icon="🛡️",
    layout="wide"
)

# --- 2. CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    [data-testid="stMetricValue"] { font-family: 'Courier New', monospace; }
    .stMetric {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
        padding: 15px !important;
    }
    .model-header { text-align: center; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
    .dt-bg { background-color: #1e3a8a; } /* Blue */
    .rf-bg { background-color: #064e3b; } /* Green */
    .knn-bg { background-color: #4c1d95; } /* Purple */
    </style>
    """, unsafe_allow_html=True)

# --- 3. RESOURCE LOADING (All 3 Models) ---
@st.cache_resource
def load_all_assets():
    try:
        dt = joblib.load('wsn_dt.pkl')
        rf = joblib.load('wsn_rf.pkl')
        knn = joblib.load('wsn_knn.pkl')
        blind_data = pd.read_csv('test_no_labels.csv')
        truth_data = pd.read_csv('test_with_labels.csv')
        return dt, rf, knn, blind_data, truth_data
    except Exception as e:
        st.error(f"⚠️ Files Missing! Run Phase 3 & 4 first. Error: {e}")
        return None, None, None, None, None

dt_model, rf_model, knn_model, blind_data, truth_data = load_all_assets()

# --- 4. SESSION STATE ---
if 'monitoring' not in st.session_state: st.session_state.monitoring = False
if 'logs' not in st.session_state: st.session_state.logs = []

# --- 5. HEADER ---
st.title("🛡️ WSN-IDS: Multi-Model Live Comparison")
st.write("Simultaneous intrusion detection using Decision Tree, Random Forest, and KNN.")

# --- 6. SIDEBAR COMMANDS ---
st.sidebar.title("🛠️ Control Panel")
if st.sidebar.button("🚀 START MONITORING", use_container_width=True):
    st.session_state.monitoring = True
if st.sidebar.button("🛑 STOP SYSTEM", use_container_width=True):
    st.session_state.monitoring = False
    st.rerun()

sim_speed = st.sidebar.slider("Scan Speed (Seconds)", 0.05, 1.0, 0.3)
st.sidebar.divider()
st.sidebar.info("This dashboard compares 16-feature blind input across 3 distinct ML architectures.")

# --- 7. LIVE COMPARISON COLUMNS ---
col_dt, col_rf, col_knn = st.columns(3)

# Placeholders for Model 1: Decision Tree
with col_dt:
    st.markdown('<div class="model-header dt-bg"><b>DECISION TREE</b></div>', unsafe_allow_html=True)
    dt_pred = st.empty()
    dt_lat = st.empty()
    dt_status = st.empty()

# Placeholders for Model 2: Random Forest
with col_rf:
    st.markdown('<div class="model-header rf-bg"><b>RANDOM FOREST</b></div>', unsafe_allow_html=True)
    rf_pred = st.empty()
    rf_lat = st.empty()
    rf_status = st.empty()

# Placeholders for Model 3: KNN
with col_knn:
    st.markdown('<div class="model-header knn-bg"><b>K-NEAREST NEIGHBORS</b></div>', unsafe_allow_html=True)
    knn_pred = st.empty()
    knn_lat = st.empty()
    knn_status = st.empty()

st.divider()

# --- 8. DATA FEED & LOGS ---
bot_left, bot_right = st.columns([2, 1])
with bot_left:
    st.subheader("📡 Live Network Packet (Blind Input)")
    packet_view = st.empty()
with bot_right:
    st.subheader("📜 Event Verification Log")
    log_view = st.empty()

# --- 9. MULTI-MODEL LOGIC ---
if st.session_state.monitoring and dt_model is not None:
    packet_count = 0
    while st.session_state.monitoring:
        packet_count += 1
        
        # 1. Pull random blind sample
        sample_row = blind_data.sample(1)
        idx = sample_row.index[0]
        actual_truth = truth_data.loc[idx, 'Attack type']

        # 2. RUN DECISION TREE
        start = time.time()
        p_dt = dt_model.predict(sample_row)[0]
        l_dt = time.time() - start

        # 3. RUN RANDOM FOREST
        start = time.time()
        p_rf = rf_model.predict(sample_row)[0]
        l_rf = time.time() - start

        # 4. RUN KNN
        start = time.time()
        p_knn = knn_model.predict(sample_row)[0]
        l_knn = time.time() - start

        # 5. UPDATE UI - DECISION TREE
        dt_pred.metric("Prediction", f"Type {p_dt}")
        dt_lat.text(f"Latency: {l_dt:.6f}s")
        if p_dt == actual_truth: dt_status.success("MATCH ✅")
        else: dt_status.error("MISS ❌")

        # 6. UPDATE UI - RANDOM FOREST
        rf_pred.metric("Prediction", f"Type {p_rf}")
        rf_lat.text(f"Latency: {l_rf:.6f}s")
        if p_rf == actual_truth: rf_status.success("MATCH ✅")
        else: rf_status.error("MISS ❌")

        # 7. UPDATE UI - KNN
        knn_pred.metric("Prediction", f"Type {p_knn}")
        knn_lat.text(f"Latency: {l_knn:.6f}s")
        if p_knn == actual_truth: knn_status.success("MATCH ✅")
        else: knn_status.error("MISS ❌")

        # 8. UPDATE FEED & LOGS
        packet_view.dataframe(sample_row)
        log_entry = f"Pkt #{packet_count} | Truth: {actual_truth} | DT: {p_dt} RF: {p_rf} KNN: {p_knn}"
        st.session_state.logs.insert(0, log_entry)
        log_view.code("\n".join(st.session_state.logs[:8]))
        
        time.sleep(sim_speed)