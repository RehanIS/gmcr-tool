'''
#####################################################################################################################################
========================================================================================================================================

THIS IS A DEPRECIATED STABLE VERSION 1.8, DO NOT USE THIS. REFER main.py!!! 

========================================================================================================================================
#####################################################################################################################################
'''



import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import os

# IMPORT OUR CUSTOM MODULES
import data_loader as dl
import ai_models as ai

# --- PAGE SETUP ---
st.set_page_config(page_title="Cohesity--GMCR", page_icon="⌂", layout="wide")

# ==========================================
# 🎨 UI VISIBILITY & STYLING
# ==========================================
st.markdown("""
    <style>
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E0E0E0;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0px 1px 2px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetric"] label {
        font-size: 14px;
        color: #666 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #000 !important;
        font-weight: 700;
    }
    /* Tabs */
    button[data-baseweb="tab"] {
        font-size: 16px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
base_dir = os.path.dirname(os.path.abspath(__file__))
logo_url = os.path.join(base_dir, 'assets', 'cohesity-logo-white-green-rgb.png')
col1, col2, col3 = st.sidebar.columns([1.05, 2, 1])
with col2:
    st.image(logo_url, width=1000)
st.sidebar.markdown("<h3 style='text-align: center;'>Cohesity--GMCR</h3>", unsafe_allow_html=True)

# 1. LOAD DATA
df_az, df_aw, df_vm = dl.load_files()
if df_az is None:
    st.error("❌ CSV files not found! Check data_loader.py paths.")
    st.stop()

# 2. CONFIGURATION
st.sidebar.header("⚙️ Configuration")
platform = st.sidebar.selectbox("Cloud Platform", ["Azure", "AWS", "VMware"])

st.sidebar.subheader("🧹 Data Refinery")
filter_range = st.sidebar.slider(
    "Data Retention Range", 0.0, 1.0, (0.01, 0.95), step=0.01,
    help="Filter out outliers (0-min errors or stuck jobs)."
)

st.sidebar.subheader("🧠 AI Engine")
model_choice = st.sidebar.radio(
    "Algorithm", 
    ["Gradient Boosting", "Random Forest", "Linear Regression", "Theoretical Physics (Formula)"]
)

# 3. TRAINING
results_dict = {}
active_scaler = None
X, y = None, None

# Select Data
if platform == 'Azure': raw_df = df_az
elif platform == 'AWS': raw_df = df_aw
else: raw_df = df_vm

region_list = raw_df['Region'].unique() if 'Region' in raw_df.columns else []

# Train Model (unless using Manual Formula)
if model_choice != "Theoretical Physics (Formula)":
    with st.spinner(f"🔄 Refining & Training {platform} Model..."):
        X, y = dl.clean_and_prep(raw_df, platform, low_q=filter_range[0], high_q=filter_range[1])
        
        @st.cache_data
        def get_training_results(X_data, y_data):
            return ai.train_all_models(X_data, y_data)
            
        results_dict, active_scaler = get_training_results(X, y)
    st.sidebar.success(f"✅ Trained on {len(X)} events")
else:
    st.sidebar.info("ℹ️ Using Static Formula")

# --- MAIN APP ---
st.image(logo_url, width=200)
st.title("📊 GMCR Prediction & Validation Tool V1.8")
tab_pred, tab_sim, tab_valid = st.tabs(["🎛️ Planner & Predictor", "🕹️ Dynamic Simulator", "🔬 Validation Lab"])

# ==========================================
# TAB 1: PREDICTOR (RESTORED DENSITY)
# ==========================================
with tab_pred:
    st.subheader(f"1. Configure Scenario ({platform})")
    
    # --- INPUT SECTION ---
    c1, c2, c3, c4, c5 = st.columns(5)
    user_vector = []
    phys_size = 0; phys_bw = 1000
    
    if platform == 'Azure':
        with c1: v1 = st.number_input("VM Size (GB)", 10, 10000, 500); phys_size = v1
        with c2: v2 = 1 if st.selectbox("Restore Method", ["Instant", "Vault"]) == 'Instant' else 0
        with c3: v3 = {'Standard_HDD':1,'Standard_SSD':2,'Premium_SSD':3,'Ultra':4}[st.selectbox("Disk Tier", ["Standard_HDD","Standard_SSD","Premium_SSD","Ultra"])]
        with c4: reg = st.selectbox("Region", region_list); v4 = 0 
        with c5: v5 = st.number_input("Bandwidth (Mbps)", 100, 10000, 1000); phys_bw = v5
        
        spd = v5 / 8.0
        t_init_phys = (v1 * 1024) / spd / 60.0 if spd > 0 else 0
        user_vector = [v1, v2, v3, v4, v5, t_init_phys]

    elif platform == 'AWS':
        with c1: v1 = st.number_input("VM Size (GB)", 10, 10000, 500); phys_size = v1
        with c2: v2 = {'st1':1, 'gp2':2, 'gp3':3, 'io2':4}[st.selectbox("Volume Type", ["st1", "gp2", "gp3", "io2"])]
        with c3: v3 = st.number_input("Provisioned IOPS", 100, 64000, 3000)
        with c4: v4 = st.number_input("Snapshot Age (Days)", 1, 365, 7)
        with c5: reg = st.selectbox("Region", region_list); v5 = 0
        
        spd = v3 * 0.25
        t_init_phys = (v1 * 1024) / spd / 60.0 if spd > 0 else 0
        user_vector = [v1, v2, v3, v4, v5, t_init_phys]

    elif platform == 'VMware':
        with c1: v1 = st.number_input("VM Size (GB)", 10, 10000, 500); phys_size = v1
        with c2: v2 = {'NBD':1, 'HotAdd':2, 'SAN':3}[st.selectbox("Transport Mode", ["NBD", "HotAdd", "SAN"])]
        with c3: v3 = st.slider("Concurrency Level", 1, 10, 4)
        with c4: v4 = st.selectbox("Network Speed (Gbps)", [1, 10, 25])
        with c5: v5 = {'HDD':1, 'SSD':3, 'NVMe':5}[st.selectbox("Target Storage", ["HDD", "SSD", "NVMe"])]
        
        spd = v4 * 125.0
        t_init_phys = (v1 * 1024) / spd / 60.0 if spd > 0 else 0
        user_vector = [v1, v2, v3, v4, v5, t_init_phys]

    st.markdown("---")

    # --- RESULTS SECTION ---
    # We split into 3 columns now to fill the space
    col_main, col_insight, col_viz = st.columns([1.2, 1.2, 2.5])
    
    # 1. CALCULATE
    final_prediction = 0.0
    confidence_score = 0.0
    
    if model_choice == "Theoretical Physics (Formula)":
        # Manual Formula
        S = phys_size; N = 4
        if platform=='VMware': N=user_vector[2]
        final_prediction = 5.0 + (0.02 * S) + (0.1 * 20) - (0.5 * N)
        if final_prediction < 1: final_prediction = 1.0
        confidence_score = 0.0 # No AI confidence
    else:
        # AI Model
        active_model_data = results_dict[model_choice]
        model = active_model_data['model_object']
        confidence_score = active_model_data['R2']
        
        scaled_input = active_scaler.transform([user_vector])
        pred_val = model.predict(scaled_input)[0]
        final_prediction = max(1.0, pred_val)

    # 2. DISPLAY - COLUMN 1 (THE NUMBERS)
    with col_main:
        st.subheader("⏱️ Forecast")
        st.metric("Final Prediction", f"{final_prediction:.1f} min")
        
        # Reality Gap (Physics vs AI)
        overhead = final_prediction - t_init_phys
        st.metric("Cloud Friction (Overhead)", f"{overhead:+.1f} min", 
                  help="The extra time caused by Cloud Latency, API throttling, and Spin-up time.",
                  delta_color="inverse")

    # 3. DISPLAY - COLUMN 2 (THE TRUST)
    with col_insight:
        st.subheader("🧠 Model Trust")
        if model_choice == "Theoretical Physics (Formula)":
             st.info("Using Fixed Formula")
             st.caption("Standard mathematical model. No confidence score available.")
        else:
            # Display Confidence Gauge
            if confidence_score > 0.8:
                st.success(f"High Confidence: {confidence_score:.1%}")
            elif confidence_score > 0.6:
                st.warning(f"Medium Confidence: {confidence_score:.1%}")
            else:
                st.error(f"Low Confidence: {confidence_score:.1%}")
                
            st.caption("How well does the AI know this scenario?")
            st.progress(max(0.0, min(1.0, confidence_score)))
            
            # Physics Comparison Text
            st.markdown(f"**Physics Baseline:** `{t_init_phys:.1f} min`")
            st.markdown(f"**AI Correction:** `{overhead/final_prediction:.0%} of total`")

    # 4. DISPLAY - COLUMN 3 (THE CHART)
    with col_viz:
        st.subheader("📈 Scalability Curve")
        # Generate Curve
        sizes = np.linspace(10, 5000, 40)
        curve_y = []
        
        if model_choice == "Theoretical Physics (Formula)":
             for s in sizes:
                val = 5.0 + (0.02 * s) + 2.0 - 2.0
                curve_y.append(max(1.0, val))
        else:
            dummy_inputs = []
            for s in sizes:
                row = list(user_vector)
                row[0] = s 
                # Re-calc physics
                if platform == 'Azure': spd = user_vector[4]/8.0
                elif platform == 'AWS': spd = user_vector[2]*0.25
                elif platform == 'VMware': spd = user_vector[3]*125.0
                if spd > 0: row[-1] = (s * 1024) / spd / 60.0
                else: row[-1] = 0
                dummy_inputs.append(row)
            
            dummy_scaled = active_scaler.transform(dummy_inputs)
            curve_y = model.predict(dummy_scaled)
            
        fig = px.line(x=sizes, y=curve_y, labels={'x':'VM Size (GB)', 'y':'Time (Min)'})
        fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=250)
        # Add a marker for current prediction
        fig.add_trace(go.Scatter(x=[phys_size], y=[final_prediction], mode='markers', marker=dict(color='red', size=10), name='Current'))
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: SIMULATOR
# ==========================================
with tab_sim:
    st.subheader("🕹️ Dynamic Recovery Simulation")
    with st.expander("ℹ️ Simulator Logic"):
        st.write("Simulates packet transfer volatility.")
        st.latex(r"T_{rem} = \frac{\sum Size_{remaining}}{Avg(Bandwidth_{t-10s} \dots Bandwidth_t)}")

    c_ctrl, c_view = st.columns([1, 2])
    with c_ctrl:
        bw = st.slider("Base Bandwidth (Mbps)", 10, 2000, 500)
        vol = st.slider("Instability", 0, 100, 20)
        start = st.button("▶️ START", type="primary")

    with c_view:
        stat = st.empty(); bar = st.progress(0)
        m1, m2 = st.columns(2)
        mt = m1.empty(); ms = m2.empty()
        
        if start:
            tot = phys_size * 1024; rem = tot
            hist = []
            while rem > 0:
                noise = np.random.randint(-vol, vol)
                cur = max(1, bw + noise)
                spd_mb = cur / 8.0
                hist.append(spd_mb)
                if len(hist)>10: hist.pop(0)
                avg = sum(hist)/len(hist)
                
                rem -= spd_mb * 0.2
                trem = (rem/avg)/60.0 if avg>0 else 999
                
                pct = max(0.0, min(1.0, 1.0 - (rem/tot)))
                bar.progress(pct)
                mt.metric("⏳ Remaining", f"{trem:.1f} min")
                ms.metric("🚀 Speed", f"{avg:.1f} MB/s")
                time.sleep(0.05)
            stat.success("DONE"); bar.progress(1.0); mt.metric("Remaining", "0.0 min")

# ==========================================
# TAB 3: VALIDATION LAB
# ==========================================
with tab_valid:
    if model_choice == "Theoretical Physics (Formula)":
        st.warning("⚠️ Manual Formula: No historical validation data available.")
    else:
        st.subheader(f"🔬 Accuracy Analysis: {model_choice}")
        val = results_dict[model_choice]['validation_data']
        yt = val['Actual']; yp = val['Predicted']
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Prediction vs Actual**")
            fig = px.scatter(x=yt, y=yp, labels={'x':'Actual','y':'Predicted'}, opacity=0.6)
            fig.add_shape(type="line", line=dict(dash='dash', color='red'), x0=yt.min(), y0=yt.max(), x1=yt.min(), y1=yt.max())
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("**Error Distribution**")
            fig = px.histogram(x=yt-yp, nbins=30, labels={'x':'Error (Min)'})
            st.plotly_chart(fig, use_container_width=True)
            
        st.markdown("**Detailed Test Data**")
        df_c = pd.DataFrame({'Actual':yt, 'Pred':yp, 'Diff':yt-yp})
        st.dataframe(df_c.head(10).style.background_gradient(subset=['Diff'], cmap='RdYlGn_r'), use_container_width=True)