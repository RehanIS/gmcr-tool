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
st.set_page_config(page_title="Cohesity--GMCR", page_icon="🛡️", layout="wide")

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

# --- SIDEBAR & LOGO LOGIC ---
base_dir = os.path.dirname(os.path.abspath(__file__))
# Attempt to find local logo, otherwise use online fallback
local_logo_path = os.path.join(base_dir, 'assets', 'cohesity-logo-white-green-rgb.png')
online_logo_url = "https://cdn-icons-png.flaticon.com/512/2331/2331966.png"

col1, col2, col3 = st.sidebar.columns([1.05, 2, 1])
with col2:
    if os.path.exists(local_logo_path):
        st.image(local_logo_path, width=150)
    else:
        st.image(online_logo_url, width=100)

st.sidebar.markdown("<h3 style='text-align: center;'>Cohesity--GMCR</h3>", unsafe_allow_html=True)

# 1. LOAD DATA
df_az, df_aw, df_vm = dl.load_files()
if df_az is None:
    st.error("❌ CSV files not found! Check 'data' folder and data_loader.py paths.")
    st.stop()

# 2. CONFIGURATION
st.sidebar.header("⚙️ Configuration")
platform = st.sidebar.selectbox("Cloud Platform", ["Azure", "AWS", "VMware"])

# --- RICH TOOLTIP: DATA REFINERY ---
refinery_help = """
**What is this?**
This control filters the historical data used to train the AI. It removes statistical anomalies to prevent the model from learning "bad habits."

**How it works:**
- **Left Handle (Low %):** Filters out "Too Good To Be True" jobs. (e.g., A restore that took 0 seconds because of a logging error).
- **Right Handle (High %):** Filters out "Catastrophic Failures." (e.g., A job that got stuck for 3 days due to a network outage).

**Impact:**
- **Narrow Range (e.g., 0.10 - 0.90):** The AI ignores extreme cases. It becomes very precise for "normal" days but might underestimate chaos.
- **Wide Range (e.g., 0.01 - 0.99):** The AI learns from almost everything. It becomes more robust but prediction variance increases.
"""

st.sidebar.subheader("🧹 Data Refinery")
filter_range = st.sidebar.slider(
    "Outlier Exclusion Filter", 
    0.0, 1.0, (0.01, 0.95), step=0.01,
    help=refinery_help
)

# --- RICH TOOLTIP: AI ENGINE ---
model_help = """
**Choose the 'Brain' of the operation:**

1. **Gradient Boosting:** The smartest option. Builds multiple decision trees, where each tree corrects the errors of the previous one. Great for complex cloud behaviors (throttling, latency spikes).
2. **Random Forest:** A democracy of decision trees. Good at handling noisy data without overfitting.
3. **Linear Regression:** A simple straight line. Useful only if you believe the relationship is purely linear (Size vs Time).
4. **Theoretical Physics:** **No AI.** Uses the raw formula:
   T = β₀ + (β₁·S) + (β₂·RTT) + (β₃·N) + (β₄·L)
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

# Select Data
if platform == 'Azure': raw_df = df_az
elif platform == 'AWS': raw_df = df_aw
else: raw_df = df_vm

region_list = raw_df['Region'].unique() if 'Region' in raw_df.columns else []

# Train Model
if model_choice != "Theoretical Physics (Formula)":
    with st.spinner(f"🔄 Refining & Training {platform} Model..."):
        X, y = dl.clean_and_prep(raw_df, platform, low_q=filter_range[0], high_q=filter_range[1])
        
        @st.cache_data
        def get_training_results(X_data, y_data):
            return ai.train_all_models(X_data, y_data)
            
        results_dict, active_scaler = get_training_results(X, y)
    st.sidebar.success(f"✅ Trained on {len(X)} verified events")
else:
    st.sidebar.info("ℹ️ Using Pure Math Mode")

# --- MAIN APP ---
st.image(local_logo_path, width=200)
st.title("📊 GMCR Prediction & Validation Tool V2.6")
tab_pred, tab_sim, tab_valid = st.tabs(["🎛️ Planner & Predictor", "🕹️ Dynamic Simulator", "🔬 Validation Lab"])

# ==========================================
# TAB 1: PREDICTOR
# ==========================================
with tab_pred:
    st.subheader(f"1. Configure Scenario ({platform})")
    
    # --- INPUT SECTION ---
    c1, c2, c3, c4, c5 = st.columns(5)
    user_vector = []
    phys_size = 0; phys_bw = 1000; reg_val = "Local"
    
    # Common Tooltips
    bw_help = "The raw allocated bandwidth. Note: Real-world TCP throughput is often 60-80% of this value due to protocol overhead."
    size_help = "Total size of the Virtual Machine or Dataset to be restored."
    
    if platform == 'Azure':
        with c1: v1 = st.number_input("VM Size (GB)", 10, 10000, 500, help=size_help); phys_size = v1
        with c2: v2 = 1 if st.selectbox("Restore Method", ["Instant", "Vault"], help="Instant mounts the image immediately; Vault copies data back to primary storage.") == 'Instant' else 0
        with c3: v3 = {'Standard_HDD':1,'Standard_SSD':2,'Premium_SSD':3,'Ultra':4}[st.selectbox("Disk Tier", ["Standard_HDD","Standard_SSD","Premium_SSD","Ultra"], help="Faster disks (SSD/Ultra) reduce write-latency during restore.")]
        with c4: reg = st.selectbox("Region", region_list, help="Geographic location. Different regions have different API latency profiles."); v4 = 0; reg_val = reg
        with c5: v5 = st.number_input("Bandwidth (Mbps)", 100, 10000, 1000, help=bw_help); phys_bw = v5
        
        spd = v5 / 8.0
        t_init_phys = (v1 * 1024) / spd / 60.0 if spd > 0 else 0
        user_vector = [v1, v2, v3, v4, v5, t_init_phys]

    elif platform == 'AWS':
        with c1: v1 = st.number_input("VM Size (GB)", 10, 10000, 500, help=size_help); phys_size = v1
        with c2: v2 = {'st1':1, 'gp2':2, 'gp3':3, 'io2':4}[st.selectbox("Volume Type", ["st1", "gp2", "gp3", "io2"], help="EBS Volume Type. 'io2' offers the highest consistency and lowest latency.")]
        with c3: v3 = st.number_input("Provisioned IOPS", 100, 64000, 3000, help="Input/Output Operations Per Second. AWS throttles speed if IOPS are too low for the data size.")
        with c4: v4 = st.number_input("Snapshot Age (Days)", 1, 365, 7, help="Older snapshots in 'Archive' tiers may take longer to hydrate before restore begins.")
        with c5: reg = st.selectbox("Region", region_list); v5 = 0; reg_val = reg
        
        spd = v3 * 0.25
        t_init_phys = (v1 * 1024) / spd / 60.0 if spd > 0 else 0
        user_vector = [v1, v2, v3, v4, v5, t_init_phys]

    elif platform == 'VMware':
        with c1: v1 = st.number_input("VM Size (GB)", 10, 10000, 500, help=size_help); phys_size = v1
        with c2: v2 = {'NBD':1, 'HotAdd':2, 'SAN':3}[st.selectbox("Transport Mode", ["NBD", "HotAdd", "SAN"], help="SAN (Storage Area Network) is fastest (direct access). NBD (Network Block Device) goes over the LAN.")]
        with c3: v3 = st.slider("Concurrency Level", 1, 10, 4, help="Number of parallel data streams. Increasing this boosts speed up to a limit, then causes congestion.")
        with c4: v4 = st.selectbox("Network Speed (Gbps)", [1, 10, 25], help="Physical link speed of the ESXi host.")
        with c5: v5 = {'HDD':1, 'SSD':3, 'NVMe':5}[st.selectbox("Target Storage", ["HDD", "SSD", "NVMe"], help="The destination datastore. NVMe absorbs data much faster than spinning HDD.")]
        
        spd = v4 * 125.0
        t_init_phys = (v1 * 1024) / spd / 60.0 if spd > 0 else 0
        user_vector = [v1, v2, v3, v4, v5, t_init_phys]

    st.markdown("---")

    # --- RESULTS SECTION ---
    col_main, col_insight, col_viz = st.columns([1.2, 1.2, 2.5])
    
    # 1. CALCULATE
    final_prediction = 0.0
    confidence_score = 0.0
    
    # [START: SPECIFIC FORMULA IMPLEMENTATION]
    if model_choice == "Theoretical Physics (Formula)":
        # Inputs
        S = phys_size
        # RTT Inference
        rtt_map = {'East US': 20, 'West US': 60, 'North Europe': 45, 'Southeast Asia': 150}
        RTT = rtt_map.get(reg_val, 30) # Default 30ms if region not found
        # Concurrency (N)
        N = user_vector[2] if platform == 'VMware' else 4
        # Load (L) - Simulated congestion factor (0.0 to 1.0)
        L = 0.5 

        # Coefficients (Betas)
        b0 = 0.5    # Intercept (Startup time in min)
        # b1 is dynamic: (Size / Bandwidth). 
        # 1 GB = 8192 Mb. If BW=1000Mbps, Time/GB = 8.192s = 0.13 min.
        b1 = (8192.0 / max(1, phys_bw)) / 60.0 
        b2 = 0.05   # RTT Penalty
        b3 = -0.1   # Concurrency Benefit
        b4 = 2.0    # Load Penalty

        # Formula: T = b0 + (b1*S) + (b2*RTT) + (b3*N) + (b4*L)
        final_prediction = b0 + (b1 * S) + (b2 * RTT) + (b3 * N) + (b4 * L)
        
        if final_prediction < 1: final_prediction = 1.0
    # [END: SPECIFIC FORMULA IMPLEMENTATION]
    
    else:
        active_model_data = results_dict[model_choice]
        model = active_model_data['model_object']
        confidence_score = active_model_data['R2']
        
        scaled_input = active_scaler.transform([user_vector])
        pred_val = model.predict(scaled_input)[0]
        final_prediction = max(1.0, pred_val)

    # 2. DISPLAY - COLUMN 1
    with col_main:
        st.subheader("⏱️ Forecast")
        st.metric("Final Prediction", f"{final_prediction:.1f} min")
        
        overhead = final_prediction - t_init_phys
        st.metric("Cloud Friction (Overhead)", f"{overhead:+.1f} min", 
                  help="This is the 'Invisible Delay'. It represents time lost to API throttling, network latency, and cloud spin-up time that pure physics formulas miss.",
                  delta_color="inverse")

    # 3. DISPLAY - COLUMN 2
    with col_insight:
        st.subheader("🧠 Model Trust")
        if model_choice == "Theoretical Physics (Formula)":
             st.info("Using Linear Regression")
             st.latex(r"T = \beta_0 + \beta_1 S + \beta_2 RTT + \beta_3 N + \beta_4 L")
             st.caption(f"Inputs: S={S}, RTT={RTT}, N={N}, L={L}")
        else:
            if confidence_score > 0.8:
                st.success(f"High Confidence: {confidence_score:.1%}")
            elif confidence_score > 0.6:
                st.warning(f"Medium Confidence: {confidence_score:.1%}")
            else:
                st.error(f"Low Confidence: {confidence_score:.1%}")
            
            st.caption("How well does the AI know this scenario?")
            st.progress(max(0.0, min(1.0, confidence_score)))
            
            st.markdown(f"**Physics Baseline:** `{t_init_phys:.1f} min`")
            st.markdown(f"**AI Correction:** `{overhead/final_prediction:.0%} of total`")

    # 4. DISPLAY - COLUMN 3
    with col_viz:
        st.subheader("📈 Scalability Curve")
        sizes = np.linspace(10, 5000, 40)
        curve_y = []
        
        if model_choice == "Theoretical Physics (Formula)":
             for s_val in sizes:
                # Re-apply exact formula for the curve
                val = b0 + (b1 * s_val) + (b2 * RTT) + (b3 * N) + (b4 * L)
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
        fig.add_trace(go.Scatter(x=[phys_size], y=[final_prediction], mode='markers', marker=dict(color='red', size=10), name='Current'))
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: SIMULATOR
# ==========================================
with tab_sim:
    st.subheader("🕹️ Dynamic Recovery Simulation")
    
    sim_help = """
    **Why use this?**
    The 'Predictor' in Tab 1 gives you the *destination* (End Time).
    This Simulator shows you the *journey* (The volatility of the transfer).
    
    **How it works:**
    We take your base bandwidth and inject random 'noise' every 0.05 seconds. 
    This mimics real-world internet instability (packet loss, jitter, shared line congestion).
    """
    
    with st.expander("ℹ️ How the Simulator Works"):
        st.write(sim_help)
        st.latex(r"T_{rem} = \frac{\sum Size_{remaining}}{Avg(Bandwidth_{t-10s} \dots Bandwidth_t)}")

    c_ctrl, c_view = st.columns([1, 2])
    with c_ctrl:
        bw = st.slider("Base Bandwidth (Mbps)", 10, 2000, 500, help="The average speed of your connection.")
        vol = st.slider("Instability (Noise)", 0, 100, 20, help="Higher noise = More fluctuating speeds. Set to 0 for a perfect, stable line.")
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
        st.warning("⚠️ Manual Formula: No historical validation data available (Because it's not an AI model).")
    else:
        st.subheader(f"🔬 Accuracy Analysis: {model_choice}")
        st.caption("This tab compares the AI's predictions against a 'Hold-Out' test set of real historical data that the AI has never seen before.")
        
        val = results_dict[model_choice]['validation_data']
        yt = val['Actual']; yp = val['Predicted']
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Prediction vs Actual**")
            fig = px.scatter(x=yt, y=yp, labels={'x':'Actual Time (Min)','y':'Predicted Time (Min)'}, opacity=0.6)
            fig.add_shape(type="line", line=dict(dash='dash', color='red'), x0=yt.min(), y0=yt.max(), x1=yt.min(), y1=yt.max())
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Perfect predictions would land exactly on the red dashed line.")
            
        with c2:
            st.markdown("**Error Distribution**")
            fig = px.histogram(x=yt-yp, nbins=30, labels={'x':'Error (Min)'})
            st.plotly_chart(fig, use_container_width=True)
            st.caption("A tall, narrow spike near 0 means the model is very accurate.")
            
        st.markdown("**Detailed Test Data**")
        df_c = pd.DataFrame({'Actual':yt, 'Pred':yp, 'Diff':yt-yp})
        st.dataframe(df_c.head(10).style.background_gradient(subset=['Diff'], cmap='RdYlGn_r'), use_container_width=True)