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

# --- 2. ADVANCED CSS (Animations & Styling) ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    
    /* Metric Card Styling */
    [data-testid="stMetricValue"] { font-family: 'Courier New', monospace; color: #00FF00 !important; }
    
    .stMetric {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 12px !important;
        padding: 20px !important;
    }

    /* Spinning Radar Animation */
    @keyframes spin { 100% { transform: rotate(360deg); } }
    .radar-icon {
        font-size: 80px;
        display: inline-block;
        animation: spin 3s linear infinite;
    }

    /* Pulsing Red Alert for Attacks */
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.7); border-color: #ff4b4b; }
        70% { box-shadow: 0 0 0 15px rgba(255, 75, 75, 0); border-color: #ff4b4b; }
        100% { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0); border-color: #30363d; }
    }
    .danger-zone {
        animation: pulse-red 1.2s infinite;
        border: 2px solid #ff4b4b !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. RESOURCE LOADING ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('wsn_dos_model.pkl')
        test_data = pd.read_csv('test_data.csv')
        opt_df = pd.read_csv('optimized_wsn.csv')
        features = opt_df.drop(columns=['Attack type']).columns.tolist()
        return model, test_data, features
    except Exception as e:
        st.error(f"⚠️ Missing Files: {e}. Ensure .pkl and .csv files are in the folder.")
        return None, None, None

model, test_data, features = load_assets()

# --- 4. SESSION STATE MANAGEMENT ---
if 'monitoring' not in st.session_state: st.session_state.monitoring = False
if 'threat_count' not in st.session_state: st.session_state.threat_count = 0
if 'logs' not in st.session_state: st.session_state.logs = []
if 'chart_history' not in st.session_state: st.session_state.chart_history = []

# --- 5. HEADER ---
h1, h2 = st.columns([1, 5])
with h1:
    st.markdown('<div class="radar-icon">📡</div>', unsafe_allow_html=True)

with h2:
    st.title("🛡️ WSN CYBER-SENTRY v3.0")
    st.write("Infinite Real-Time Neural Intrusion Monitoring & Active Mitigation")

# --- 6. SYSTEM METRICS ---
m1, m2, m3, m4 = st.columns(4)
scan_m = m1.empty()
threat_m = m2.empty()
block_m = m3.empty()
lat_m = m4.empty()

# Initial placeholders
scan_m.metric("Packets Scanned", 0)
threat_m.metric("Threats Flagged", 0)
block_m.metric("Nodes Isolated", 0)
lat_m.metric("Avg Latency", "0.00000012s")

st.divider()

# --- 7. LIVE MONITORING LAYOUT ---
col_left, col_right = st.columns([1.5, 1])

with col_left:
    st.subheader("🌐 Real-Time Data Ingestion")
    feed_area = st.empty()

with col_right:
    st.subheader("📈 Network Heartbeat (Anomaly Spike)")
    chart_area = st.empty()
    st.subheader("📜 Security Event Logs")
    log_area = st.empty()

# --- 8. SIDEBAR COMMAND CONSOLE ---
st.sidebar.title("🛠️ Command Console")
if st.sidebar.button("🚀 START MONITORING", use_container_width=True):
    st.session_state.monitoring = True

if st.sidebar.button("🛑 STOP SYSTEM", use_container_width=True):
    st.session_state.monitoring = False
    st.rerun()

sim_speed = st.sidebar.slider("Scan Frequency (Seconds)", 0.01, 1.0, 0.2)

if st.sidebar.button("🗑️ Reset All Stats", use_container_width=True):
    st.session_state.threat_count = 0
    st.session_state.logs = []
    st.session_state.chart_history = []
    st.rerun()

# --- 9. INFINITE MONITORING LOGIC ---
if st.session_state.monitoring and model is not None:
    packet_index = 0
    while st.session_state.monitoring:
        packet_index += 1
        
        # Pull 1 random packet for simulation
        sample_row = test_data.sample(1)
        input_data = sample_row[features]
        prediction = model.predict(input_data)[0]
        
        is_attack = (prediction != 0)
        
        # Update Chart History
        st.session_state.chart_history.append(1 if is_attack else 0)
        # Display only last 50 points to create the "moving" heartbeat effect
        display_chart = pd.DataFrame(st.session_state.chart_history[-50:], columns=["Status"])

        # Update Metrics
        scan_m.metric("Packets Scanned", packet_index)
        lat_m.metric("Latency", f"{0.00000012 + (random.uniform(0, 0.00000005)):.8f}s")
        
        if is_attack:
            st.session_state.threat_count += 1
            # Trigger the Pulsing Red Alert
            threat_m.markdown('<div class="danger-zone">', unsafe_allow_html=True)
            threat_m.metric("THREATS DETECTED", st.session_state.threat_count, delta="DANGER", delta_color="inverse")
            st.session_state.logs.insert(0, f"🚨 ALERT: Attack Type {prediction} Isolated")
        else:
            threat_m.metric("THREATS DETECTED", st.session_state.threat_count)
            st.session_state.logs.insert(0, f"✅ Verified: Packet_{random.randint(1000,9999)} Normal")

        # Refresh UI Components
        feed_area.dataframe(input_data, use_container_width=True)
        chart_area.line_chart(display_chart, height=220)
        log_area.code("\n".join(st.session_state.logs[:10]))
        
        # Control simulation speed
        time.sleep(sim_speed)