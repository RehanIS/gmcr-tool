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
st.set_page_config(page_title="GMCR--Recovery", page_icon="🛡️", layout="wide")

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
    /* Success/Warning Messages */
    .stAlert {
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR & LOGO LOGIC ---
st.sidebar.markdown("<h2 style='text-align: center;'>GMCR</h2>", unsafe_allow_html=True)
st.sidebar.caption("Global Multi-Cloud Recovery")

# 1. LOAD DATA
try:
    pass 
except Exception as e:
    st.error(f"Connection Error: {e}")

# 2. CONFIGURATION
st.sidebar.header("⚙️ Configuration")
platform = st.sidebar.selectbox("Cloud Platform", ["Azure", "AWS", "VMware"])

# Load Data for Selected Platform
raw_df = dl.fetch_training_data(platform)

if raw_df.empty:
    st.sidebar.warning("⚠️ No data found in DB. Using default/empty set.")
    raw_df = pd.DataFrame(columns=['vm_size_gb', 'restore_time_min'])

# --- RICH TOOLTIP: DATA REFINERY ---
refinery_help = """
**What is this?**
Filters historical data to remove anomalies.
- **Left Handle:** Filters "Too Good To Be True" (0s restores).
- **Right Handle:** Filters "Catastrophic Failures" (Outages).
"""

st.sidebar.subheader("🧹 Data Refinery")
filter_range = st.sidebar.slider(
    "Outlier Exclusion Filter", 
    0.0, 1.0, (0.01, 0.95), step=0.01,
    help=refinery_help
)

# --- RICH TOOLTIP: AI ENGINE ---
model_help = """
**Choose the 'Brain':**
1. **Gradient Boosting:** Smartest. Corrects previous errors.
2. **Random Forest:** Good for noisy data.
3. **Linear Regression:** Simple straight line.
4. **Theoretical Physics:** Raw formula (No AI).
"""

st.sidebar.subheader("🧠 AI Engine")
model_choice = st.sidebar.radio(
    "Prediction Algorithm", 
    ["Gradient Boosting", "Random Forest", "Linear Regression", "Theoretical Physics (Formula)"],
    help=model_help
)

# 3. TRAINING
results_dict = {}
active_scaler = None
X, y = None, None
region_list = []

# Prepare Data
if not raw_df.empty:
    if 'region' in raw_df.columns:
        region_list = raw_df['region'].unique()
    elif 'Region' in raw_df.columns:
        region_list = raw_df['Region'].unique()
    
    # Train Model
    if model_choice != "Theoretical Physics (Formula)":
        with st.spinner(f"🔄 Refining & Training {platform} Model..."):
            # Clean & Prep
            X, y = dl.clean_and_prep(raw_df, platform, low_q=filter_range[0], high_q=filter_range[1])
            
            if X is not None and len(X) > 10:
                # Unpack 3 values: Model, Scaler, Metrics
                m_obj, m_scaler, m_metrics = ai.train_model(X, y, model_choice, platform)
                
                # STORE EVERYTHING (Model, Scaler, Metrics, AND Data for Validation Tab)
                results_dict[model_choice] = {
                    'model': m_obj,
                    'scaler': m_scaler,
                    'metrics': m_metrics,
                    'X': X,
                    'y': y
                }
                active_scaler = m_scaler 
                
                st.sidebar.success(f"✅ Trained on {len(X)} verified events")
            else:
                st.sidebar.warning("⚠️ Not enough data to train AI.")
    else:
        st.sidebar.info("ℹ️ Using Pure Math Mode")

# --- MAIN APP ---
st.title("📊 GMCR Prediction & Validation Tool V2.7")
tab_pred, tab_sim, tab_valid, tab_feed = st.tabs(["🎛️ Planner", "🕹️ Simulator", "🔬 Validation", "🗣️ Feedback Loop"])

# ==========================================
# TAB 1: PREDICTOR
# ==========================================
with tab_pred:
    st.subheader(f"1. Configure Scenario ({platform})")
    
    # --- INPUT SECTION ---
    c1, c2, c3, c4, c5 = st.columns(5)
    user_vector = []
    phys_size = 0; phys_bw = 1000
    
    # Common Tooltips
    bw_help = "The raw allocated bandwidth."
    size_help = "Total size of the Virtual Machine or Dataset."
    
    # ---------------- PREDICTION INPUTS ----------------
    if platform == 'Azure':
        with c1: 
            v1 = st.number_input("VM Size (GB)", 10, 10000, 500, help=size_help)
            phys_size = v1
        with c2: 
            method_name = st.selectbox("Restore Method", ["Instant", "Vault"])
            v2 = 1 if method_name == 'Instant' else 0
        with c3:
            if method_name == "Instant":
                tier_name = st.selectbox("Disk Tier", ["Standard_HDD","Standard_SSD","Premium_SSD","Ultra"])
                v3 = {'Standard_HDD':1,'Standard_SSD':2,'Premium_SSD':3,'Ultra':4}[tier_name]
            else:
                redundancy_name = st.selectbox("Vault Redundancy", ["LRS", "GRS", "RA-GRS"])
                v3 = {'LRS':1, 'GRS':2, 'RA-GRS':3}[redundancy_name]
        with c4: 
            reg = st.selectbox("Region", region_list if len(region_list)>0 else ["East US"])
            v4 = 0 
        with c5: 
            v5 = st.number_input("Bandwidth (Mbps)", 100, 10000, 1000, help=bw_help)
            phys_bw = v5
        
        spd = v5 / 8.0
        t_init_phys = (v1 * 1024) / spd / 60.0 if spd > 0 else 0
        user_vector = [v1, v2, v3, v4, v5, t_init_phys]

    elif platform == 'AWS':
        with c1: 
            v1 = st.number_input("VM Size (GB)", 10, 10000, 500, help=size_help)
            phys_size = v1
        with c2: 
            vol_name = st.selectbox("Volume Type", ["st1", "gp2", "gp3", "io2"])
            v2 = {'st1':1, 'gp2':2, 'gp3':3, 'io2':4}[vol_name]
        with c3: 
            v3 = st.number_input("Provisioned IOPS", 100, 64000, 3000)
        with c4: 
            v4 = st.number_input("Snapshot Age (Days)", 1, 365, 7)
        with c5: 
            reg = st.selectbox("Region", region_list if len(region_list)>0 else ["us-east-1"])
            v5 = 0
        
        spd = v3 * 0.25
        t_init_phys = (v1 * 1024) / spd / 60.0 if spd > 0 else 0
        user_vector = [v1, v2, v3, v4, v5, t_init_phys]

    elif platform == 'VMware':
        with c1: 
            v1 = st.number_input("VM Size (GB)", 10, 10000, 500, help=size_help)
            phys_size = v1
        with c2: 
            mode_name = st.selectbox("Transport Mode", ["NBD", "HotAdd", "SAN"])
            v2 = {'NBD':1, 'HotAdd':2, 'SAN':3}[mode_name]
        with c3: 
            v3 = st.slider("Concurrency Level", 1, 10, 4)
        with c4: 
            v4 = st.selectbox("Network Speed (Gbps)", [1, 10, 25])
        with c5: 
            store_name = st.selectbox("Target Storage", ["HDD", "SSD", "NVMe"])
            v5 = {'HDD':1, 'SSD':3, 'NVMe':5}[store_name]
        
        spd = v4 * 125.0
        t_init_phys = (v1 * 1024) / spd / 60.0 if spd > 0 else 0
        user_vector = [v1, v2, v3, v4, v5, t_init_phys]

    st.markdown("---")

    # --- RESULTS SECTION ---
    col_main, col_insight, col_viz = st.columns([1.2, 1.2, 2.5])
    
    # 1. CALCULATE
    final_prediction = 0.0
    confidence_score = 0.0
    
    if model_choice == "Theoretical Physics (Formula)":
        S = phys_size
        RTT = 30
        N = user_vector[2] if platform == 'VMware' else 4
        L = 0.5 
        b0 = 0.5 
        b1 = (8192.0 / max(1, phys_bw)) / 60.0 
        b2 = 0.05 
        b3 = -0.1 
        b4 = 2.0 
        final_prediction = max(1.0, b0 + (b1 * S) + (b2 * RTT) + (b3 * N) + (b4 * L))
    
    else:
        if active_scaler and model_choice in results_dict:
            # RETRIEVE FROM DICTIONARY
            res = results_dict[model_choice]
            active_model = res['model']
            metrics = res['metrics']
            confidence_score = metrics['R2']
            
            scaled_input = active_scaler.transform([user_vector])
            pred_val = active_model.predict(scaled_input)[0]
            final_prediction = max(1.0, pred_val)

    # 2. DISPLAY - COLUMN 1 (FORECAST)
    with col_main:
        st.subheader("⏱️ Forecast")
        st.metric("Final Prediction", f"{final_prediction:.1f} min")
        
        overhead = final_prediction - t_init_phys
        st.metric("Cloud Friction (Overhead)", f"{overhead:+.1f} min", 
                 delta_color="inverse" if overhead > 0 else "normal")

    # 3. DISPLAY - COLUMN 2 (MODEL TRUST)
    with col_insight:
        st.subheader("🧠 Model Trust")
        
        if model_choice == "Theoretical Physics (Formula)":
             st.info("Using Linear Regression Formula")
        else:
            # Color-coded Confidence Badge
            if confidence_score > 0.8: 
                st.success(f"**High Confidence:** {confidence_score:.1%}")
            elif confidence_score > 0.6: 
                st.warning(f"**Medium Confidence:** {confidence_score:.1%}")
            else: 
                st.error(f"**Low Confidence:** {confidence_score:.1%}")
            
            st.progress(max(0.0, min(1.0, confidence_score)))
            
            # AI CORRECTION METRIC
            st.caption(f"Physics Baseline: **{t_init_phys:.1f} min**")
            
            if t_init_phys > 0:
                correction_pct = ((final_prediction - t_init_phys) / t_init_phys) * 100
                color_hex = "#4CAF50" if abs(correction_pct) < 20 else "#FF9800"
                st.markdown(f"AI Correction: <span style='color:{color_hex}'>**{correction_pct:+.0f}%** of total</span>", unsafe_allow_html=True)

    # 4. DISPLAY - COLUMN 3 (VIZ)
    with col_viz:
        st.subheader("📈 Scalability Curve")
        sizes = np.linspace(10, 5000, 40)
        curve_y = []
        
        if model_choice == "Theoretical Physics (Formula)":
             for s_val in sizes:
                val = b0 + (b1 * s_val) + (b2 * RTT) + (b3 * N) + (b4 * L)
                curve_y.append(max(1.0, val))
        else:
            if active_scaler and model_choice in results_dict:
                dummy_inputs = []
                active_model = results_dict[model_choice]['model']
                
                for s in sizes:
                    row = list(user_vector)
                    row[0] = s 
                    # Re-calc physics feature
                    if platform == 'Azure': spd = user_vector[4]/8.0
                    elif platform == 'AWS': spd = user_vector[2]*0.25
                    elif platform == 'VMware': spd = user_vector[3]*125.0
                    
                    if spd > 0: row[-1] = (s * 1024) / spd / 60.0
                    else: row[-1] = 0
                    dummy_inputs.append(row)
                
                dummy_scaled = active_scaler.transform(dummy_inputs)
                curve_y = active_model.predict(dummy_scaled)
            
        fig = px.line(x=sizes, y=curve_y, labels={'x':'VM Size (GB)', 'y':'Time (Min)'})
        fig.update_layout(margin=dict(l=20, r=20, t=10, b=20), height=220)
        fig.add_trace(go.Scatter(x=[phys_size], y=[final_prediction], mode='markers', marker=dict(color='red', size=12), name='Current'))
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: SIMULATOR
# ==========================================
with tab_sim:
    st.subheader("🕹️ Dynamic Recovery Simulation")
    
    with st.expander("ℹ️ How the Simulator Works"):
        st.write("Simulates network noise and packet loss volatility.")

    c_ctrl, c_view = st.columns([1, 2])
    with c_ctrl:
        bw = st.slider("Base Bandwidth (Mbps)", 10, 2000, 500)
        vol = st.slider("Instability (Noise)", 0, 100, 20)
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
        st.warning("⚠️ Manual Formula: No validation data available.")
    elif model_choice not in results_dict:
        st.info("Train a model to see metrics.")
    else:
        # RETRIEVE STORED DATA
        res = results_dict[model_choice]
        model = res['model']
        scaler = res['scaler']
        metrics = res['metrics']
        X_val = res['X']
        y_val = res['y']
        
        st.subheader(f"🔬 Accuracy Analysis: {model_choice}")
        
        # 1. RE-RUN PREDICTIONS FOR VIZ
        X_scaled = scaler.transform(X_val)
        y_pred = model.predict(X_scaled)
        
        # 2. CHARTS ROW
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("**Prediction vs Actual**")
            fig_scat = px.scatter(x=y_val, y=y_pred, labels={'x': 'Actual Time (Min)', 'y': 'Predicted Time (Min)'})
            # Add a "Perfect Line"
            fig_scat.add_shape(type="line", x0=0, y0=0, x1=max(y_val), y1=max(y_val), 
                               line=dict(color="Red", width=2, dash="dash"))
            st.plotly_chart(fig_scat, use_container_width=True)
            
        with c2:
            st.markdown("**Error Distribution**")
            error = y_pred - y_val
            fig_hist = px.histogram(x=error, nbins=30, labels={'x': 'Error (Min)'})
            st.plotly_chart(fig_hist, use_container_width=True)

        # 3. DETAILED DATA TABLE (Traffic Light Style)
        st.markdown("### Detailed Test Data")
        st.caption("Perfect predictions would land exactly on the red dashed line.")
        
        df_results = pd.DataFrame({
            'Actual': y_val,
            'Pred': y_pred,
            'Diff': error
        }).head(20) # Limit to top 20 for readability
        
        # Function to color code the table
        def color_diff(val):
            color = '#4CAF50' if abs(val) < 5 else '#FF9800' if abs(val) < 20 else '#f44336'
            return f'color: {color}; font-weight: bold'

        st.dataframe(df_results.style.map(color_diff, subset=['Diff']), use_container_width=True)

# ==========================================
# TAB 4: FEEDBACK LOOP
# ==========================================
with tab_feed:
    st.subheader("🗣️ Improve the Model")
    st.markdown("Did you perform a real restore? **Enter the actual details below** to make the AI smarter.")
    
    with st.form("feedback_form"):
        st.markdown(f"**Platform:** {platform}")
        
        # --- REPLICATE INPUTS FOR FEEDBACK ONLY ---
        c1, c2, c3, c4 = st.columns(4)
        
        feedback_inputs = {}
        
        # ---------------- AZURE FEEDBACK ----------------
        if platform == 'Azure':
            with c1: f_v1 = st.number_input("VM Size (GB)", 10, 10000, 100, key="f_az_sz")
            with c2: 
                f_method = st.selectbox("Restore Method", ["Instant", "Vault"], key="f_az_mth")
            with c3:
                if f_method == "Instant":
                    f_param = st.selectbox("Disk Tier", ["Standard_HDD","Standard_SSD","Premium_SSD","Ultra"], key="f_az_tier")
                else:
                    f_param = st.selectbox("Vault Redundancy", ["LRS", "GRS", "RA-GRS"], key="f_az_red")
            with c4: f_reg = st.selectbox("Region", region_list if len(region_list)>0 else ["East US"], key="f_az_reg")
            f_bw = st.number_input("Bandwidth (Mbps)", 100, 10000, 1000, key="f_az_bw")

            feedback_inputs = {
                'size': f_v1, 'method': f_method, 'tier_or_red': f_param, 'region': f_reg, 'bw': f_bw
            }

        # ---------------- AWS FEEDBACK ----------------
        elif platform == 'AWS':
            with c1: f_v1 = st.number_input("VM Size (GB)", 10, 10000, 100, key="f_aw_sz")
            with c2: f_vol = st.selectbox("Volume Type", ["st1", "gp2", "gp3", "io2"], key="f_aw_vol")
            with c3: f_iops = st.number_input("Provisioned IOPS", 100, 64000, 3000, key="f_aw_iops")
            with c4: f_age = st.number_input("Snapshot Age (Days)", 1, 365, 7, key="f_aw_age")
            f_reg = st.selectbox("Region", region_list if len(region_list)>0 else ["us-east-1"], key="f_aw_reg")

            feedback_inputs = {
                'size': f_v1, 'vol_type': f_vol, 'iops': f_iops, 'age': f_age, 'region': f_reg
            }

        # ---------------- VMWARE FEEDBACK ----------------
        elif platform == 'VMware':
            with c1: f_v1 = st.number_input("VM Size (GB)", 10, 10000, 100, key="f_vm_sz")
            with c2: f_mode = st.selectbox("Transport Mode", ["NBD", "HotAdd", "SAN"], key="f_vm_mod")
            with c3: f_conc = st.slider("Concurrency Level", 1, 10, 4, key="f_vm_conc")
            with c4: f_net = st.selectbox("Network Speed (Gbps)", [1, 10, 25], key="f_vm_net")
            f_store = st.selectbox("Target Storage", ["HDD", "SSD", "NVMe"], key="f_vm_store")

            feedback_inputs = {
                'size': f_v1, 'mode': f_mode, 'concurrency': f_conc, 'net_gbps': f_net, 'storage': f_store
            }

        st.markdown("---")
        
        # ACTUAL TIME INPUT
        real_act = st.number_input("Actual Time Taken (min)", min_value=1.0, value=10.0, step=0.1)
        
        submitted = st.form_submit_button("💾 Submit Feedback")
        
        if submitted:
            # Call data_loader to save
            success = dl.save_feedback(platform, feedback_inputs, real_act)
            if success:
                st.success("✅ Feedback saved! The model will learn from this in the next training cycle.")
                time.sleep(2)
                st.rerun()
            else:
                st.error("❌ Failed to save feedback. Check database connection.")