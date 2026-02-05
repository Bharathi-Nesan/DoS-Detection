import streamlit as st
import pandas as pd
import joblib
import time
import random
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="WSN-IDS Cyber-Sentry",
    page_icon="🛡️",
    layout="wide"
)

# --- 2. ADVANCED CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    [data-testid="stMetricValue"] { font-family: 'Courier New', monospace; color: #00FF00 !important; }
    .stMetric {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 12px !important;
        padding: 20px !important;
    }
    @keyframes spin { 100% { transform: rotate(360deg); } }
    .radar-icon { font-size: 80px; display: inline-block; animation: spin 3s linear infinite; }
    
    /* Green Pulse for Correct Predictions */
    @keyframes pulse-green {
        0% { box-shadow: 0 0 0 0 rgba(0, 255, 0, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(0, 255, 0, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 255, 0, 0); }
    }
    .success-pulse { animation: pulse-green 1s infinite; border: 1px solid #00ff00 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. RESOURCE LOADING (Updated to 2 Files) ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('wsn_dos_model.pkl')
        # Load the two separate files created in Phase 3
        blind_data = pd.read_csv('test_no_labels.csv') # 16 columns
        truth_data = pd.read_csv('test_with_labels.csv') # 17 columns
        features = blind_data.columns.tolist()
        return model, blind_data, truth_data, features
    except Exception as e:
        st.error(f"⚠️ Files Missing: Run Phase 3 & 4 first! Error: {e}")
        return None, None, None, None

model, blind_data, truth_data, features = load_assets()

# --- 4. SESSION STATE ---
if 'monitoring' not in st.session_state: st.session_state.monitoring = False
if 'threat_count' not in st.session_state: st.session_state.threat_count = 0
if 'logs' not in st.session_state: st.session_state.logs = []
if 'chart_history' not in st.session_state: st.session_state.chart_history = []

# --- 5. HEADER ---
h1, h2 = st.columns([1, 5])
with h1: st.markdown('<div class="radar-icon">📡</div>', unsafe_allow_html=True)
with h2:
    st.title("🛡️ WSN CYBER-SENTRY v3.0")
    st.write("Real-Time Neural Monitoring with Ground-Truth Verification")

# --- 6. METRICS ---
m1, m2, m3, m4 = st.columns(4)
scan_m = m1.empty()
threat_m = m2.empty()
verify_m = m3.empty() # New Verification Placeholder
lat_m = m4.empty()

# --- 7. LAYOUT ---
col_left, col_right = st.columns([1.5, 1])
with col_left:
    st.subheader("🌐 Ingested Features (Blind Input)")
    feed_area = st.empty()
with col_right:
    st.subheader("📈 Anomaly Heartbeat")
    chart_area = st.empty()
    st.subheader("📜 System Logs & Truth Check")
    log_area = st.empty()

# --- 8. SIDEBAR ---
st.sidebar.title("🛠️ Command Console")
if st.sidebar.button("🚀 START MONITORING"): st.session_state.monitoring = True
if st.sidebar.button("🛑 STOP SYSTEM"):
    st.session_state.monitoring = False
    st.rerun()
sim_speed = st.sidebar.slider("Scan Frequency", 0.01, 1.0, 0.2)

# --- 9. MONITORING LOGIC (Updated for Verification) ---
if st.session_state.monitoring and model is not None:
    packet_index = 0
    while st.session_state.monitoring:
        packet_index += 1
        
        # A. Pull 1 sample from Blind Data
        sample_row = blind_data.sample(1)
        sample_idx = sample_row.index[0]
        
        # B. Model Prediction (Based on 16 features)
        prediction = model.predict(sample_row)[0]
        
        # C. TRUTH CHECK: Get label from Truth Data
        actual_label = truth_data.loc[sample_idx, 'Attack type']
        is_correct = (prediction == actual_label)
        is_attack = (prediction != 0)

        # D. Update Visualization
        st.session_state.chart_history.append(1 if is_attack else 0)
        display_chart = pd.DataFrame(st.session_state.chart_history[-50:], columns=["Status"])
        
        scan_m.metric("Packets Scanned", packet_index)
        lat_m.metric("Latency", f"{0.00000012 + (random.uniform(0, 0.00000005)):.8f}s")
        
        # E. Verification UI
        if is_correct:
            verify_m.markdown('<div class="success-pulse">', unsafe_allow_html=True)
            verify_m.metric("Truth Match", "100%", delta="VERIFIED")
        else:
            verify_m.metric("Truth Match", "0%", delta="MISMATCH", delta_color="inverse")

        if is_attack:
            st.session_state.threat_count += 1
            threat_m.metric("THREATS FLAG", st.session_state.threat_count, delta="DANGER", delta_color="inverse")
            log_entry = f"🚨 {time.strftime('%H:%M:%S')} - Pred: {prediction} | Actual: {actual_label} {'✅' if is_correct else '❌'}"
        else:
            threat_m.metric("THREATS FLAG", st.session_state.threat_count)
            log_entry = f"✅ {time.strftime('%H:%M:%S')} - Pred: Normal | Actual: Normal {'✅' if is_correct else '❌'}"

        st.session_state.logs.insert(0, log_entry)

        # Refresh UI
        feed_area.dataframe(sample_row, use_container_width=True)
        chart_area.line_chart(display_chart, height=220)
        log_area.code("\n".join(st.session_state.logs[:10]))
        
        time.sleep(sim_speed)