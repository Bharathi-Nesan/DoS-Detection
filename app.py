import streamlit as st
import pandas as pd
import joblib
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- 1. ADMIN & EMAIL CONFIGURATION ---
ADMIN_EMAIL = "kokila2732007@gmail.com"      
SENDER_ACCOUNT = "bharathinesan2k6@gmail.com"   
APP_PASSWORD = "aqmb gjmv bdvs lfiv"      

# Attack Mapping Dictionary
ATTACK_MAP = {
    0: "Normal",
    1: "Blackhole",
    2: "Grayhole",
    3: "Flooding",
    4: "TDMA Scheduling"
}

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(page_title="WSN-IDS Multi-Model Sentry", page_icon="🛡️", layout="wide")

# --- 3. EMAIL NOTIFICATION FUNCTION ---
def send_admin_alert(attack_name, model_name):
    try:
        msg = MIMEMultipart()
        # This line sets the "System Name" that the Admin sees
        msg['From'] = f"🛡️ WSN-IDS SENTRY <{SENDER_ACCOUNT}>" 
        msg['To'] = ADMIN_EMAIL
        msg['Subject'] = f"🚨 SYSTEM ALERT: {attack_name} Detected"
        
        body = (f"--- AUTOMATED WSN SECURITY REPORT ---\n\n"
               f"THREAT LEVEL: CRITICAL\n"
               f"Detected Attack: {attack_name}\n"
               f"Monitoring Model: {model_name}\n"
               f"Timestamp: {time.ctime()}\n\n"
               f"Action Required: Please log in to the secure dashboard to mitigate the threat.")
        
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_ACCOUNT, APP_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Email Error: {e}")
        return False

# --- 4. CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    [data-testid="stMetricValue"] { font-family: 'Courier New', monospace; font-size: 24px !important; }
    .stMetric { background: #161b22 !important; border: 1px solid #30363d !important; border-radius: 10px !important; padding: 15px !important; }
    .model-header { text-align: center; padding: 10px; border-radius: 5px; margin-bottom: 10px; font-weight: bold; }
    .dt-bg { background-color: #1e3a8a; } 
    .rf-bg { background-color: #064e3b; } 
    .knn-bg { background-color: #4c1d95; } 
    </style>
    """, unsafe_allow_html=True)

# --- 5. RESOURCE LOADING ---
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
        st.error(f"⚠️ Files Missing! Error: {e}")
        return None, None, None, None, None

dt_model, rf_model, knn_model, blind_data, truth_data = load_all_assets()

# --- 6. SESSION STATE ---
if 'monitoring' not in st.session_state: st.session_state.monitoring = False
if 'logs' not in st.session_state: st.session_state.logs = []
if 'last_email_time' not in st.session_state: st.session_state.last_email_time = 0

# --- 7. HEADER & SIDEBAR ---
st.title("🛡️ WSN-IDS: Multi-Model Live Comparison")
if st.sidebar.button("🚀 START MONITORING", use_container_width=True):
    st.session_state.monitoring = True
if st.sidebar.button("🛑 STOP SYSTEM", use_container_width=True):
    st.session_state.monitoring = False
    st.rerun()
sim_speed = st.sidebar.slider("Scan Speed", 0.05, 1.0, 0.3)

# --- 8. LIVE COMPARISON UI ---
col_dt, col_rf, col_knn = st.columns(3)
with col_dt:
    st.markdown('<div class="model-header dt-bg">DECISION TREE</div>', unsafe_allow_html=True)
    dt_pred = st.empty(); dt_lat = st.empty(); dt_status = st.empty()
with col_rf:
    st.markdown('<div class="model-header rf-bg">RANDOM FOREST</div>', unsafe_allow_html=True)
    rf_pred = st.empty(); rf_lat = st.empty(); rf_status = st.empty()
with col_knn:
    st.markdown('<div class="model-header knn-bg">KNN</div>', unsafe_allow_html=True)
    knn_pred = st.empty(); knn_lat = st.empty(); knn_status = st.empty()

st.divider()
bot_left, bot_right = st.columns([2, 1])
with bot_left:
    st.subheader("📡 Live Network Packet")
    packet_view = st.empty()
with bot_right:
    st.subheader("📜 Event Log")
    log_view = st.empty()

# --- 9. MONITORING LOGIC ---
if st.session_state.monitoring and dt_model is not None:
    packet_count = 0
    while st.session_state.monitoring:
        packet_count += 1
        sample_row = blind_data.sample(1)
        idx = sample_row.index[0]
        actual_val = truth_data.loc[idx, 'Attack type']
        actual_name = ATTACK_MAP.get(actual_val, "Unknown")

        # Inference
        t1 = time.time(); p_dt_val = dt_model.predict(sample_row)[0]; l_dt = time.time() - t1
        t2 = time.time(); p_rf_val = rf_model.predict(sample_row)[0]; l_rf = time.time() - t2
        t3 = time.time(); p_knn_val = knn_model.predict(sample_row)[0]; l_knn = time.time() - t3

        # Convert numbers to names for UI
        n_dt = ATTACK_MAP.get(p_dt_val, "Unknown")
        n_rf = ATTACK_MAP.get(p_rf_val, "Unknown")
        n_knn = ATTACK_MAP.get(p_knn_val, "Unknown")

        # Update Metrics
        dt_pred.metric("Prediction", n_dt); dt_lat.text(f"Lat: {l_dt:.6f}s")
        dt_status.success("MATCH ✅") if p_dt_val == actual_val else dt_status.error("MISS ❌")
        
        rf_pred.metric("Prediction", n_rf); rf_lat.text(f"Lat: {l_rf:.6f}s")
        rf_status.success("MATCH ✅") if p_rf_val == actual_val else rf_status.error("MISS ❌")
        
        knn_pred.metric("Prediction", n_knn); knn_lat.text(f"Lat: {l_knn:.6f}s")
        knn_status.success("MATCH ✅") if p_knn_val == actual_val else knn_status.error("MISS ❌")

        # TRIGGER EMAIL (30s Cooldown)
        if p_dt_val != 0:
            current_time = time.time()
            if (current_time - st.session_state.last_email_time) > 30:
                if send_admin_alert(n_dt, "Pruned Decision Tree"):
                    st.toast(f"📧 Email Sent to {ADMIN_EMAIL}!", icon="📩")
                    st.session_state.last_email_time = current_time

        packet_view.dataframe(sample_row)
        log_entry = f"Pkt #{packet_count} | Truth: {actual_name} | DT:{n_dt}"
        st.session_state.logs.insert(0, log_entry)
        log_view.code("\n".join(st.session_state.logs[:8]))
        time.sleep(sim_speed)