import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk # NEW: For 3D Global Mapping
import time
import os
from datetime import datetime
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import report_generator as rg

# IMPORT OUR CUSTOM MODULES
import data_loader as dl
import ai_models as ai

# --- PAGE SETUP ---
st.set_page_config(page_title="GMCR Orchestrator", page_icon="🛡️", layout="wide")

# ==========================================
# 📝 DETAILED TOOLTIP DEFINITIONS
# ==========================================
tooltips = {
    "forecast": """
    **Final Estimated Restore Time.** This is the AI's definitive prediction for how long the operation will take to complete. 
    It is calculated by combining the theoretical Physics Baseline (Size ÷ Bandwidth) with historical 
    patterns learned from previous real-world restores. This value accounts for hidden variables 
    such as cloud API latency, disk hydration time, TCP handshakes, and regional network congestion 
    that simple mathematical formulas often miss.
    """,
    
    "friction": """
    **Cloud Infrastructure Overhead.** This metric isolates the "lost time" caused purely by cloud inefficiencies. 
    A positive value (e.g., +12.5 min) indicates delays caused by throttling, resource provisioning 
    (waiting for Azure/AWS to allocate the VM), or storage I/O bottlenecks. 
    Essentially, this is the difference between "Physics" (theoretical speed) and "Reality." 
    If this number is high, consider upgrading your Disk Tier or Provisioned IOPS.
    """,
    
    "trust": """
    **AI Reliability Score (R²).** This percentage indicates how well the AI's training data matches your current specific scenario. 
    - **High (>80%):** The model has seen very similar restores before; trust this prediction.
    - **Low (<60%):** The model is guessing based on limited data. Rely more on the Physics Baseline.
    
    This section also includes the **AI Correction Factor**, which quantifies exactly how much the AI 
    had to adjust the raw math to match reality. A high correction factor means your environment 
    is behaving significantly differently than the hardware datasheet suggests.
    """,
    
    "curve": """
    **Data Scalability Projection.** This chart visualizes how the restore time increases as the Data Size (GB) grows. 
    - **Linear (Straight Line):** Healthy. Your infrastructure scales perfectly with data size.
    - **Exponential (Curved Up):** Unhealthy. Indicates a bottleneck (like IOPS limits or Bandwidth saturation) 
    where adding more data causes disproportionately longer delays. 
    
    Use this curve to find the "Tipping Point" where your current architecture becomes inefficient.
    """,
    
    "sim_main": """
    **Monte Carlo Network Simulation.** Unlike the static prediction in Tab 1, this Simulator mimics the chaotic nature of a real-world network. 
    It introduces random variables like packet loss, jitter, and bandwidth fluctuations (Noise) to show you 
    how the restore might behave under stress. 
    
    Use this to stress-test your Recovery Time Objective (RTO) against worst-case network conditions.
    """,
    
    "sim_bw": """
    **Average Sustained Throughput.** The baseline speed you expect from your ISP or Cloud Direct Connect link. 
    The simulator treats this as the "Mean" speed. In reality, your speed will fluctuate 
    above and below this number based on the Instability setting. 
    Set this to your guaranteed Committed Information Rate (CIR).
    """,
    
    "sim_vol": """
    **Network Volatility (Standard Deviation).** Controls how "noisy" the connection is. 
    - **Low (0-10):** Represents a dedicated Dark Fiber or ExpressRoute link (very stable).
    - **High (50+):** Represents a public internet connection or VPN with heavy contention and packet loss. 
    
    Higher instability leads to unpredictable completion times and "micro-stalls" during the restore.
    """
}

# ==========================================
# 🎨 UI VISIBILITY & STYLING
# ==========================================
st.markdown("""
    <style>
    /* Premium Enterprise Dark Mode Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #1A1C23 !important;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] label {
        font-size: 14px;
        color: #888 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #00F0FF !important;
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
st.sidebar.markdown("<h2 style='text-align: center; color: #00F0FF;'>GMCR Engine</h2>", unsafe_allow_html=True)
st.sidebar.caption("Global Multi-Cloud Recovery Orchestrator")

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
        region_list = sorted(raw_df['region'].astype(str).unique())
    elif 'Region' in raw_df.columns:
        region_list = sorted(raw_df['Region'].astype(str).unique())
    
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
st.title("🛡️ DR Orchestrator & IaC Generator")
tab_pred, tab_iac, tab_sim, tab_valid, tab_hist, tab_compare, tab_report, tab_feed = st.tabs([
    "🎛️ AI Orchestrator", 
    "🏗️ Architecture & IaC", 
    "🕹️ Chaos Simulator", 
    "🔬 Validation & Explainability", 
    "📊 Historical Trends",
    "⚖️ Multi-Cloud & Cost Analysis",
    "📄 Executive Report",
    "🗣️ Feedback Loop"
])

# Global variables for cross-tab sharing
phys_size = 500
phys_bw = 1000
tier_name = "Standard"
vol_name = "gp3"
reg = "East US"
v3 = 3000
t_init_phys = 0

# ==========================================
# TAB 1: AI ORCHESTRATOR
# ==========================================
with tab_pred:
    st.subheader(f"1. Configure Workload Payload ({platform})")
    
    # --- INPUT SECTION ---
    c1, c2, c3, c4, c5 = st.columns(5)
    user_vector = []
    
    # Common Tooltips (Inputs)
    bw_help = "The raw allocated bandwidth/throughput available for this restore job."
    size_help = "Total size of the Virtual Machine or Dataset to be restored."
    reg_help = "Geographic location. Crossing regions adds significant latency (RTT)."
    
    # ---------------- PREDICTION INPUTS ----------------
    if platform == 'Azure':
        with c1: 
            v1 = st.number_input("VM Size (GB)", 10, 10000, 500, help=size_help)
            phys_size = v1
        with c2: 
            method_name = st.selectbox("Restore Method", ["Instant", "Vault"], help="Instant mounts the image immediately; Vault copies data back (slower).")
            v2 = 1 if method_name == 'Instant' else 0
        with c3:
            if method_name == "Instant":
                tier_name = st.selectbox("Disk Tier", ["Standard_HDD","Standard_SSD","Premium_SSD","Ultra"], help="Faster disks (SSD/Ultra) reduce write-latency.")
                v3 = {'Standard_HDD':1,'Standard_SSD':2,'Premium_SSD':3,'Ultra':4}[tier_name]
            else:
                redundancy_name = st.selectbox("Vault Redundancy", ["LRS", "GRS", "RA-GRS"], help="Geo-Redundant (GRS) takes longer to verify/retrieve than Local (LRS).")
                v3 = {'LRS':1, 'GRS':2, 'RA-GRS':3}[redundancy_name]
                tier_name = redundancy_name
        with c4: 
            r_list = region_list if len(region_list) > 0 else ["East US", "West Europe", "SE Asia", "West US"]
            reg = st.selectbox("Region", r_list, help=reg_help)
            
            if reg in r_list:
                v4 = r_list.index(reg) + 1
            else:
                v4 = 1 # Fallback
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
            vol_name = st.selectbox("Volume Type", ["st1", "gp2", "gp3", "io2"], help="EBS Volume Type. 'io2/gp3' offer higher throughput than 'st1'.")
            v2 = {'st1':1, 'gp2':2, 'gp3':3, 'io2':4}[vol_name]
            tier_name = vol_name
        with c3: 
            v3 = st.number_input("Provisioned IOPS", 100, 64000, 3000, help="AWS throttles speed if IOPS are too low for the data size.")
        with c4: 
            v4 = st.number_input("Snapshot Age (Days)", 1, 365, 7, help="Older snapshots (Archive tier) require 'hydration' time before restore starts.")
        with c5: 
            r_list = region_list if len(region_list) > 0 else ["us-east-1", "us-west-1", "eu-central-1"]
            reg = st.selectbox("Region", r_list, help=reg_help)
            
            if reg in r_list:
                v5 = r_list.index(reg) + 1
            else:
                v5 = 1 # Fallback
        
        spd = v3 * 0.25
        t_init_phys = (v1 * 1024) / spd / 60.0 if spd > 0 else 0
        phys_bw = v3 * 0.25 * 8 # Approximation for mapping logic
        user_vector = [v1, v2, v3, v4, v5, t_init_phys]

    elif platform == 'VMware':
        with c1: 
            v1 = st.number_input("VM Size (GB)", 10, 10000, 500, help=size_help)
            phys_size = v1
        with c2: 
            mode_name = st.selectbox("Transport Mode", ["NBD", "HotAdd", "SAN"], help="SAN (Storage Area Network) is fastest (direct). NBD is over LAN (slowest).")
            v2 = {'NBD':1, 'HotAdd':2, 'SAN':3}[mode_name]
        with c3: 
            v3 = st.slider("Concurrency Level", 1, 10, 4, help="Number of parallel streams. Too high = congestion.")
        with c4: 
            v4 = st.selectbox("Network Speed (Gbps)", [1, 10, 25], help="Physical link speed of the ESXi host.")
            phys_bw = v4 * 1000
        with c5: 
            store_name = st.selectbox("Target Storage", ["HDD", "SSD", "NVMe"], help="Write speed of destination. NVMe absorbs data fastest.")
            v5 = {'HDD':1, 'SSD':3, 'NVMe':5}[store_name]
            tier_name = store_name
        
        spd = v4 * 125.0
        t_init_phys = (v1 * 1024) / spd / 60.0 if spd > 0 else 0
        user_vector = [v1, v2, v3, v4, v5, t_init_phys]

    st.markdown("---")

    # --- RESULTS SECTION ---
    # 1. CALCULATE
    final_prediction = 0.0
    confidence_score = 0.0
    
    if model_choice == "Theoretical Physics (Formula)":
        S = phys_size
        region_idx = user_vector[3] if platform == 'Azure' else (user_vector[4] if platform == 'AWS' else 1)
        RTT = 20 + (region_idx * 5) 
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
            res = results_dict[model_choice]
            active_model = res['model']
            metrics = res['metrics']
            confidence_score = metrics['R2']
            
            scaled_input = active_scaler.transform([user_vector])
            pred_val = active_model.predict(scaled_input)[0]
            final_prediction = max(1.0, pred_val)

    # 2. FINOPS & METRICS ROW (NEW)
    col_main, col_finops = st.columns([1.2, 2])
    
    with col_main:
        st.subheader("⏱️ AI Forecast", help=tooltips['forecast'])
        st.metric("Final Prediction", f"{final_prediction:.1f} min")
        
        overhead = final_prediction - t_init_phys
        st.metric("Cloud Friction (Overhead)", f"{overhead:+.1f} min", 
                 delta_color="inverse" if overhead > 0 else "normal",
                 help=tooltips['friction'])

    with col_finops:
        st.subheader("💼 FinOps & Business Impact Analysis")
        fin_c1, fin_c2 = st.columns(2)
        
        with fin_c1:
            sla_target = st.number_input("Target SLA (Minutes)", min_value=1, value=60)
        with fin_c2:
            downtime_cost = st.number_input("Cost of Downtime ($/Hour)", min_value=1000, value=25000, step=1000)
        
        st.markdown("<br>", unsafe_allow_html=True)
        is_breach = final_prediction > sla_target
        
        if is_breach:
            breach_mins = final_prediction - sla_target
            financial_loss = (breach_mins / 60) * downtime_cost
            st.error(f"🚨 **SLA BREACH DETECTED**")
            st.markdown(f"**Projected Loss:** <span style='color:#FF5252; font-size:20px; font-weight:bold'>${financial_loss:,.2f}</span>", unsafe_allow_html=True)
            
            # Actionable Intelligence
            st.warning("⚠️ **AI Mitigation Recommendation**")
            if phys_bw < 2000:
                st.markdown("👉 **Bottleneck:** Network Bandwidth is saturating. Upgrade base bandwidth.")
            elif platform == 'Azure' and (tier_name == 'Standard_HDD' or tier_name == 'Standard_SSD'):
                st.markdown(f"👉 **Bottleneck:** Target Storage I/O. Provision Premium_SSD or Ultra Disks.")
            elif platform == 'AWS' and v3 < 5000:
                st.markdown(f"👉 **Bottleneck:** Throttled Provisioned IOPS. Upgrade to io2 Block Express.")
            else:
                st.markdown("👉 **Bottleneck:** Data Gravity. Break dataset into parallel recovery shards.")
        else:
            buffer = sla_target - final_prediction
            st.success(f"✅ **SLA SECURE**")
            st.markdown(f"**Safety Buffer:** <span style='color:#4CAF50; font-size:20px; font-weight:bold'>{buffer:.1f} mins</span>", unsafe_allow_html=True)

    st.markdown("---")

    # 3. SCALABILITY CURVE & MODEL TRUST (RESTORED FROM ORIGINAL)
    col_insight, col_viz = st.columns([1.2, 2.5])

    with col_insight:
        st.subheader("🧠 Model Trust", help=tooltips['trust'])
        
        if model_choice == "Theoretical Physics (Formula)":
             st.info("Using Linear Regression Formula")
        else:
            if confidence_score > 0.8: 
                st.success(f"**High Confidence:** {confidence_score:.1%}")
            elif confidence_score > 0.6: 
                st.warning(f"**Medium Confidence:** {confidence_score:.1%}")
            else: 
                st.error(f"**Low Confidence:** {confidence_score:.1%}")
            
            st.progress(max(0.0, min(1.0, confidence_score)))
            
            st.caption(f"Physics Baseline: **{t_init_phys:.1f} min**", help=tooltips['trust'])
            
            if t_init_phys > 0:
                correction_pct = ((final_prediction - t_init_phys) / t_init_phys) * 100
                color_hex = "#4CAF50" if abs(correction_pct) < 20 else "#FF9800"
                st.markdown(f"AI Correction: <span style='color:{color_hex}'>**{correction_pct:+.0f}%** of total</span>", unsafe_allow_html=True)

    with col_viz:
        st.subheader("📈 Scalability Curve", help=tooltips['curve'])
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

    st.markdown("---")

    # 4. 3D GLOBAL ROUTING MAP (NEW)
    st.subheader("🌐 Global Recovery Telemetry", help="Live 3D routing from the Primary Vault to the target edge region.")
    
    region_coords = {
        "East US": [37.3719, -79.8164], "West US": [37.7749, -122.4194],
        "West Europe": [52.3676, 4.9041], "SE Asia": [1.3521, 103.8198],
        "us-east-1": [38.1300, -78.4500], "eu-central-1": [50.1109, 8.6821],
        "Default": [0, 0]
    }
    
    target_coord = region_coords.get(reg, region_coords["East US"])
    vault_coord = [41.8781, -87.6298] # Chicago
    
    map_data = pd.DataFrame({
        "start_lat": [vault_coord[0]], "start_lon": [vault_coord[1]],
        "end_lat": [target_coord[0]], "end_lon": [target_coord[1]],
        "name": [reg]
    })

    st.pydeck_chart(pdk.Deck(
        map_style="dark",
        initial_view_state=pdk.ViewState(
            latitude=30.0,
            longitude=-40.0,
            zoom=1.5,
            pitch=45,
        ),
        layers=[
            pdk.Layer(
                "ArcLayer",
                data=map_data,
                get_source_position="[start_lon, start_lat]",
                get_target_position="[end_lon, end_lat]",
                get_source_color=[0, 255, 0, 160], 
                get_target_color=[255, 0, 0, 160], 
                auto_highlight=True,
                width_scale=0.1,
                get_width="10 * 10",
                width_min_pixels=3,
                width_max_pixels=30,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=map_data,
                get_position="[end_lon, end_lat]",
                get_color=[255, 0, 0, 200],
                get_radius=200000,
            ),
        ],
    ))

# ==========================================
# TAB 2: ARCHITECTURE & IaC (NEW)
# ==========================================
with tab_iac:
    st.subheader("🏗️ Autonomous Infrastructure Orchestration")
    st.markdown("Why manually open ports on a VPS? Let the AI generate the exact infrastructure code needed to deploy the optimized recovery environment securely and at scale.")

    top_col, sre_col = st.columns([1, 1])

    with top_col:
        st.markdown("### 🕸️ Generated DR Topology")
        icon = "☁️ AWS" if platform == "AWS" else "☁️ Azure" if platform == "Azure" else "☁️ VMware"
        graph_code = f"""
        digraph G {{
            rankdir=LR; bgcolor="transparent";
            node [shape=box, style=filled, color="#1A1C23", fontcolor="#00F0FF", fontname="Helvetica"]; edge [color="#E0E0E0"];
            Vault [label="System Vault", shape=cylinder, color="#0E1117"]; WAF [label="Zero-Trust FW\n(Port 443)"];
            VPC [label="{icon} Clean Room\n({reg})"]; App [label="Recovery Node\n{phys_size}GB"]; Storage [label="Storage\n{tier_name}", shape=cylinder];
            Vault -> WAF [label=" Encrypted", fontcolor="#00F0FF"]; WAF -> VPC -> App -> Storage;
        }}
        """
        st.graphviz_chart(graph_code)

    with sre_col:
        st.markdown("### 🛤️ SRE Critical Path Sequencer")
        st.info("Eliminates boot-storm race conditions via staggered orchestration.")
        t_db = final_prediction
        t_mid = final_prediction * 0.4
        t_front = final_prediction * 0.2
        st.markdown(f"1. 🗄️ **Database:** `{t_db:.1f} mins` *(Blocks Tier 2)*")
        st.markdown(f"2. ⚙️ **Middleware:** `{t_mid:.1f} mins` *(Starts after DB)*")
        st.markdown(f"3. 🌐 **Frontend/WAF:** `{t_front:.1f} mins` *(Starts after Middleware)*")
        st.progress(100, text=f"Total Enterprise App RTO: {(t_db+t_mid+t_front):.1f} mins")

    st.markdown("---")
    
    st.markdown("### ⚡ Zero-Touch Code Generators")
    iac_col1, iac_col2 = st.columns(2)
    
    with iac_col1:
        if st.button("🚀 Generate Terraform (VPS Setup)", use_container_width=True):
            tf_code = f"""# Automatically Generated by GMCR AI Orchestrator
# Target RTO: {final_prediction:.1f} minutes | Size: {phys_size} GB

provider "{"aws" if platform == 'AWS' else "azurerm"}" {{
  region = "{reg}"
}}

# 1. Isolated Recovery VPC
resource "{"aws_vpc" if platform == 'AWS' else "azurerm_virtual_network"}" "recovery_vpc" {{
  {"cidr_block" if platform == 'AWS' else "address_space"} = "10.0.0.0/16"
}}

# 2. Automated Firewall (Zero-Trust)
resource "{"aws_security_group" if platform == 'AWS' else "azurerm_network_security_group"}" "zero_trust_sg" {{
  name = "gmcr_recovery_sg"
  
  ingress {{
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/8"]
  }}
}}

# 3. Provisioning the VPS & Mounting Storage
resource "{"aws_instance" if platform == 'AWS' else "azurerm_linux_virtual_machine"}" "recovery_node" {{
  instance_type = "optimized_compute"
  
  # Attached High-Speed Storage
  storage_tier = "{tier_name}"
  volume_size  = {phys_size}
}}"""
            st.code(tf_code, language="hcl")
            st.download_button("⬇️ Download main.tf", data=tf_code, file_name="main.tf", mime="text/plain")
            
    with iac_col2:
        if st.button("☸️ Generate Kubernetes Manifest", use_container_width=True):
            k8s_yaml = f"""# Automatically Generated by GMCR AI Orchestrator
# Target RTO: {final_prediction:.1f} mins | Platform: {platform}

apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: System-recovery-pvc
  namespace: dr-clean-room
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: {tier_name.lower()}-class
  resources:
    requests:
      storage: {phys_size}Gi
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: gmcr-recovery-workload
spec:
  replicas: 3 # Auto-scaled across cluster
  template:
    spec:
      containers:
      - name: recovery-agent
        image: System/agent:latest
        ports:
        - containerPort: 443
        volumeMounts:
        - name: recovery-storage
          mountPath: /data
      volumes:
      - name: recovery-storage
        persistentVolumeClaim:
          claimName: System-recovery-pvc
"""
            st.code(k8s_yaml, language="yaml")
            st.download_button("⬇️ Download manifest.yaml", data=k8s_yaml, file_name="dr-manifest.yaml", mime="text/yaml")


# ==========================================
# TAB 3: SIMULATOR (ORIGINAL)
# ==========================================
with tab_sim:
    st.subheader("🕹️ Dynamic Recovery Simulation", help=tooltips['sim_main'])
    
    with st.expander("ℹ️ How the Simulator Works"):
        st.write("Simulates network noise and packet loss volatility.")

    c_ctrl, c_view = st.columns([1, 2])
    with c_ctrl:
        bw = st.slider("Base Bandwidth (Mbps)", 10, 2000, 500, help=tooltips['sim_bw'])
        vol = st.slider("Instability (Noise)", 0, 100, 20, help=tooltips['sim_vol'])
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
# TAB 4: VALIDATION LAB (ORIGINAL RESTORED)
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
            fig_scat.add_shape(type="line", x0=0, y0=0, x1=max(y_val), y1=max(y_val), line=dict(color="Red", width=2, dash="dash"))
            st.plotly_chart(fig_scat, use_container_width=True)
            
        with c2:
            st.markdown("**Error Distribution**")
            error = y_pred - y_val
            fig_hist = px.histogram(x=error, nbins=30, labels={'x': 'Error (Min)'})
            st.plotly_chart(fig_hist, use_container_width=True)

        # 3. DETAILED DATA TABLE
        st.markdown("### Detailed Test Data")
        st.caption("Perfect predictions would land exactly on the red dashed line.")
        
        df_results = pd.DataFrame({
            'Actual': y_val,
            'Pred': y_pred,
            'Diff': error
        }).head(20)
        
        def color_diff(val):
            color = '#4CAF50' if abs(val) < 5 else '#FF9800' if abs(val) < 20 else '#f44336'
            return f'color: {color}; font-weight: bold'

        st.dataframe(df_results.style.map(color_diff, subset=['Diff']), use_container_width=True)

    # ==========================================
    # SHAP EXPLAINABILITY (MERGED INTO VALIDATION TAB)
    # ==========================================
    st.markdown("---")
    st.subheader("🔍 Feature Explainability (SHAP Analysis)")
    st.markdown("Understand **which features** drive the AI prediction and by how much.")

    feature_name_map = {
        'Azure': ['VM Size (GB)', 'Restore Method', 'Disk Tier', 'Region', 'Bandwidth (Mbps)', 'Physics Baseline'],
        'AWS': ['VM Size (GB)', 'Volume Type', 'IOPS', 'Snapshot Age', 'Region', 'Physics Baseline'],
        'VMware': ['VM Size (GB)', 'Transport Mode', 'Concurrency', 'Network Speed', 'Target Storage', 'Physics Baseline']
    }
    f_names = feature_name_map.get(platform, ['F1','F2','F3','F4','F5','F6'])

    if model_choice in ["Gradient Boosting", "Random Forest"] and model_choice in results_dict:
        res_shap = results_dict[model_choice]
        shap_model = res_shap['model']
        shap_scaler = res_shap['scaler']

        # --- 1. Feature Importance Bar Chart (sklearn) ---
        st.markdown("#### Feature Importance (Model-Intrinsic)")
        importances = shap_model.feature_importances_
        imp_df = pd.DataFrame({'Feature': f_names, 'Importance': importances}).sort_values('Importance', ascending=True)
        fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                         color='Importance', color_continuous_scale='Tealgrn')
        fig_imp.update_layout(template='plotly_dark', height=300, margin=dict(l=10,r=10,t=10,b=10),
                              coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)

        # --- 2. SHAP Waterfall Chart ---
        st.markdown("#### SHAP Waterfall (Current Prediction Breakdown)")
        st.caption("Shows how each feature pushes the prediction higher or lower from the base value.")
        try:
            explainer = shap.TreeExplainer(shap_model)
            scaled_input = shap_scaler.transform([user_vector])
            shap_values = explainer(scaled_input)
            shap_values.feature_names = f_names

            fig_wf, ax_wf = plt.subplots(figsize=(10, 4))
            shap.plots.waterfall(shap_values[0], show=False)
            fig_wf = plt.gcf()
            st.pyplot(fig_wf, use_container_width=True)
            plt.close('all')

            # --- 3. SHAP Summary for training data ---
            st.markdown("#### SHAP Feature Impact (Global View)")
            X_shap = res_shap['X']
            X_shap_scaled = shap_scaler.transform(X_shap)
            shap_vals_all = explainer(X_shap_scaled)
            shap_vals_all.feature_names = f_names

            fig_bee, ax_bee = plt.subplots(figsize=(10, 4))
            shap.plots.beeswarm(shap_vals_all, show=False)
            fig_bee = plt.gcf()
            st.pyplot(fig_bee, use_container_width=True)
            plt.close('all')
        except Exception as e_shap:
            st.warning(f"SHAP waterfall could not be generated: {e_shap}")
            st.info("Feature importance bar chart above is still valid.")
    elif model_choice == "Linear Regression" and model_choice in results_dict:
        st.info("Linear Regression does not support SHAP TreeExplainer. Use Gradient Boosting or Random Forest for full explainability.")
        # Show coefficients instead
        res_lr = results_dict[model_choice]
        lr_model = res_lr['model']
        coefs = lr_model.coef_
        coef_df = pd.DataFrame({'Feature': f_names, 'Coefficient': coefs}).sort_values('Coefficient', ascending=True)
        fig_coef = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h',
                          color='Coefficient', color_continuous_scale='RdYlGn')
        fig_coef.update_layout(template='plotly_dark', height=300, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_coef, use_container_width=True)
    else:
        st.info("Train a model (Gradient Boosting or Random Forest recommended) to see SHAP explainability analysis.")

# ==========================================
# TAB 5: FEEDBACK LOOP (ORIGINAL RESTORED)
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

# ==========================================
# TAB 6: HISTORICAL TRENDS (NEW)
# ==========================================
with tab_hist:
    st.subheader("📊 Historical Restore Performance Trends")
    st.markdown("Track restore performance over time. Detect drift, validate SLA compliance, and identify seasonal patterns.")
    
    hist_df = dl.fetch_historical_trend(platform)
    
    if hist_df.empty or len(hist_df) < 2:
        st.warning(f"⚠️ Not enough historical data for **{platform}**. Submit feedback entries or add training data to see trends.")
    else:
        hist_sla = st.number_input("SLA Threshold (minutes)", min_value=1, value=60, key="hist_sla")
        
        # Time-series chart
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=hist_df['created_at'], y=hist_df['restore_time_min'],
            mode='markers+lines', name='Restore Time',
            line=dict(color='#00F0FF', width=1),
            marker=dict(size=4, color='#00F0FF', opacity=0.7)
        ))
        
        # Rolling average
        window = min(7, len(hist_df))
        hist_df['rolling_avg'] = hist_df['restore_time_min'].rolling(window=window, min_periods=1).mean()
        fig_trend.add_trace(go.Scatter(
            x=hist_df['created_at'], y=hist_df['rolling_avg'],
            mode='lines', name=f'{window}-Point Rolling Avg',
            line=dict(color='#FF9800', width=2, dash='dash')
        ))
        
        # SLA threshold line
        fig_trend.add_hline(y=hist_sla, line_dash="dot", line_color="red",
                            annotation_text=f"SLA Target ({hist_sla} min)")
        
        fig_trend.update_layout(
            xaxis_title="Date", yaxis_title="Restore Time (minutes)",
            height=420, template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Summary metrics
        st.markdown("### 📈 Performance Summary")
        hm1, hm2, hm3, hm4 = st.columns(4)
        with hm1:
            st.metric("Total Records", f"{len(hist_df):,}")
        with hm2:
            st.metric("Avg Restore Time", f"{hist_df['restore_time_min'].mean():.1f} min")
        with hm3:
            st.metric("Median", f"{hist_df['restore_time_min'].median():.1f} min")
        with hm4:
            breach_count = len(hist_df[hist_df['restore_time_min'] > hist_sla])
            breach_pct = (breach_count / len(hist_df)) * 100
            st.metric("SLA Breach Rate", f"{breach_pct:.0f}%")
        
        # Distribution
        st.markdown("---")
        hd1, hd2 = st.columns(2)
        with hd1:
            st.markdown("#### ⏱️ Restore Time Distribution")
            fig_dist = px.histogram(hist_df, x='restore_time_min', nbins=25,
                                    labels={'restore_time_min': 'Restore Time (min)'},
                                    color_discrete_sequence=['#00F0FF'])
            fig_dist.add_vline(x=hist_sla, line_dash="dot", line_color="red",
                               annotation_text="SLA")
            fig_dist.update_layout(template='plotly_dark', height=300,
                                   margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with hd2:
            st.markdown("#### 🗂️ Data Source Breakdown")
            if 'record_source' in hist_df.columns:
                source_counts = hist_df['record_source'].value_counts()
                fig_src = px.pie(values=source_counts.values, names=source_counts.index,
                                 color_discrete_sequence=['#00F0FF', '#FF9800', '#4CAF50'])
                fig_src.update_layout(template='plotly_dark', height=300,
                                      margin=dict(l=10,r=10,t=30,b=10))
                st.plotly_chart(fig_src, use_container_width=True)
            else:
                st.info("No source metadata available.")
        
        # VM Size vs Restore Time scatter
        if 'vm_size_gb' in hist_df.columns:
            st.markdown("#### 📦 VM Size vs Restore Time Correlation")
            fig_corr = px.scatter(hist_df, x='vm_size_gb', y='restore_time_min',
                                  labels={'vm_size_gb': 'VM Size (GB)', 'restore_time_min': 'Restore Time (min)'},
                                  trendline='ols', color_discrete_sequence=['#00F0FF'])
            fig_corr.update_layout(template='plotly_dark', height=300,
                                   margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_corr, use_container_width=True)


# ==========================================
# TAB 7: MULTI-CLOUD & COST ANALYSIS (NEW)
# ==========================================
with tab_compare:
    st.subheader("⚖️ Multi-Cloud Comparison & Cost Analysis")
    st.markdown("Compare restore performance and estimated cost across **Azure, AWS, and VMware** for the same workload.")
    
    st.markdown("### 🔧 Workload Specification")
    cc1, cc2 = st.columns(2)
    with cc1:
        compare_size = st.number_input("VM Size (GB)", 10, 10000, 500, key="cmp_size")
    with cc2:
        compare_bw = st.number_input("Available Bandwidth (Mbps)", 100, 10000, 1000, key="cmp_bw")
    
    st.markdown("---")
    
    # --- PHYSICS-BASED PREDICTIONS ---
    # Azure
    az_spd = compare_bw / 8.0
    az_phys = (compare_size * 1024) / az_spd / 60.0 if az_spd > 0 else 0
    az_compute = (az_phys / 60) * 0.096
    az_storage = compare_size * 0.13 / 30
    az_egress = compare_size * 0.087
    az_total = az_compute + az_storage + az_egress
    
    # AWS (default gp3, 3000 IOPS)
    aws_spd = 3000 * 0.25
    aws_phys = (compare_size * 1024) / aws_spd / 60.0 if aws_spd > 0 else 0
    aws_compute = (aws_phys / 60) * 0.096
    aws_storage = compare_size * 0.08 / 30
    aws_egress = compare_size * 0.09
    aws_total = aws_compute + aws_storage + aws_egress
    
    # VMware (default 10 Gbps)
    vmw_spd = 10 * 125.0
    vmw_phys = (compare_size * 1024) / vmw_spd / 60.0 if vmw_spd > 0 else 0
    vmw_compute = 0.0  # On-prem (amortized)
    vmw_storage = compare_size * 0.10 / 30
    vmw_egress = 0.0
    vmw_total = vmw_compute + vmw_storage + vmw_egress
    
    # --- COMPARISON TABLE ---
    st.markdown("### 📊 Side-by-Side Comparison")
    compare_df = pd.DataFrame({
        'Metric': ['Physics RTO (min)', 'Compute Cost', 'Storage Cost', 'Egress/Transfer Cost', 'Total Cost'],
        'Azure': [f"{az_phys:.1f}", f"${az_compute:.2f}", f"${az_storage:.2f}", f"${az_egress:.2f}", f"${az_total:.2f}"],
        'AWS': [f"{aws_phys:.1f}", f"${aws_compute:.2f}", f"${aws_storage:.2f}", f"${aws_egress:.2f}", f"${aws_total:.2f}"],
        'VMware': [f"{vmw_phys:.1f}", f"${vmw_compute:.2f}", f"${vmw_storage:.2f}", f"${vmw_egress:.2f}", f"${vmw_total:.2f}"]
    })
    st.dataframe(compare_df.set_index('Metric'), use_container_width=True)
    
    st.caption("💡 Azure/AWS costs include compute ($/hr), managed storage ($/GB/mo prorated), and egress. VMware is on-prem (no egress, amortized compute).")
    
    st.markdown("---")
    rd1, rd2 = st.columns([1.2, 1])
    
    with rd1:
        # --- RADAR CHART ---
        st.markdown("### 🕸️ Platform Capability Radar")
        max_phys = max(az_phys, aws_phys, vmw_phys, 1)
        max_cost = max(az_total, aws_total, vmw_total, 0.01)
        
        categories = ['Speed', 'Cost Efficiency', 'Scalability', 'Flexibility', 'Enterprise Readiness']
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[max(0, 1 - az_phys/max_phys), max(0, 1 - az_total/max_cost), 0.80, 0.70, 0.90],
            theta=categories, fill='toself', name='Azure',
            line=dict(color='#0078D4')
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[max(0, 1 - aws_phys/max_phys), max(0, 1 - aws_total/max_cost), 0.90, 0.85, 0.85],
            theta=categories, fill='toself', name='AWS',
            line=dict(color='#FF9900')
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[max(0, 1 - vmw_phys/max_phys), max(0, 1 - vmw_total/max_cost), 0.60, 0.50, 0.70],
            theta=categories, fill='toself', name='VMware',
            line=dict(color='#76B900')
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            template='plotly_dark', height=420,
            margin=dict(l=40, r=40, t=30, b=30)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with rd2:
        # --- STACKED COST BAR ---
        st.markdown("### 💰 Cost Breakdown")
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Bar(
            name='Compute', x=['Azure', 'AWS', 'VMware'],
            y=[az_compute, aws_compute, vmw_compute],
            marker_color='#00F0FF'
        ))
        fig_cost.add_trace(go.Bar(
            name='Storage', x=['Azure', 'AWS', 'VMware'],
            y=[az_storage, aws_storage, vmw_storage],
            marker_color='#FF9800'
        ))
        fig_cost.add_trace(go.Bar(
            name='Egress', x=['Azure', 'AWS', 'VMware'],
            y=[az_egress, aws_egress, vmw_egress],
            marker_color='#F44336'
        ))
        fig_cost.update_layout(
            barmode='stack', template='plotly_dark', height=420,
            yaxis_title='Cost ($)', xaxis_title='Platform',
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_cost, use_container_width=True)
    
    # --- DB Historical Stats ---
    st.markdown("---")
    st.markdown("### 📈 Historical Performance from Database")
    with st.spinner("Fetching cross-platform data..."):
        summary = dl.fetch_all_platforms_summary()
    
    sum_df = pd.DataFrame(summary).T
    sum_df.index.name = "Platform"
    if not sum_df.empty and sum_df['count'].sum() > 0:
        sum_df['count'] = sum_df['count'].astype(int)
        sum_df = sum_df.rename(columns={
            'count': 'Records', 'mean': 'Avg (min)', 'median': 'Median (min)',
            'min': 'Min (min)', 'max': 'Max (min)', 'std': 'Std Dev'
        })
        st.dataframe(sum_df, use_container_width=True)
    else:
        st.info("No historical data found in the database.")


# ==========================================
# TAB 8: EXECUTIVE REPORT (NEW)
# ==========================================
with tab_report:
    st.subheader("📄 Executive Report Generator")
    st.markdown("Generate a professional **PDF report** of the current DR analysis for stakeholders, auditors, and leadership.")
    
    rpt_c1, rpt_c2 = st.columns([1.2, 1])
    
    with rpt_c1:
        st.markdown("### 📋 Report Contents")
        st.markdown("""
        1. 📖 **Cover Page** — Platform, date, AI engine
        2. 📊 **Executive Summary** — SLA status, financial impact
        3. 🔧 **Prediction Details** — RTO, physics baseline, cloud friction
        4. ⚙️ **Workload Configuration** — All input parameters
        5. 💰 **Cost Estimation** — Compute, storage, egress breakdown
        6. 💡 **AI Recommendations** — Automated mitigation steps
        7. ⚠️ **Disclaimer** — Compliance-safe language
        """)
    
    with rpt_c2:
        st.markdown("### ⚙️ Report Parameters")
        rpt_sla = st.number_input("SLA Target (min)", min_value=1, value=60, key="rpt_sla")
        rpt_cost = st.number_input("Downtime Cost ($/hr)", min_value=1000, value=25000, step=1000, key="rpt_cost")
        
        # Show current prediction summary
        st.markdown("---")
        st.markdown("**Current Analysis:**")
        st.markdown(f"- **Platform:** {platform}")
        st.markdown(f"- **Predicted RTO:** {final_prediction:.1f} min")
        st.markdown(f"- **Physics Baseline:** {t_init_phys:.1f} min")
        is_rpt_breach = final_prediction > rpt_sla
        if is_rpt_breach:
            st.error(f"🚨 SLA Breach: {final_prediction:.1f} > {rpt_sla} min")
        else:
            st.success(f"✅ SLA Secure (buffer: {rpt_sla - final_prediction:.1f} min)")
    
    st.markdown("---")
    
    # Build config dict
    workload_config = {
        "Platform": platform,
        "AI Model": model_choice,
        "VM Size": f"{phys_size} GB",
        "Bandwidth": f"{phys_bw} Mbps",
        "Storage Tier": tier_name,
        "Region": reg,
    }
    
    # Cost data
    rpt_compute_cost = (final_prediction / 60) * 0.10
    rpt_storage_cost = phys_size * 0.10 / 30
    rpt_egress_cost = phys_size * 0.087 if platform != 'VMware' else 0
    rpt_total = rpt_compute_cost + rpt_storage_cost + rpt_egress_cost
    
    cost_data = {
        "Compute Cost (restore duration)": f"${rpt_compute_cost:.2f}",
        "Storage Cost (prorated /day)": f"${rpt_storage_cost:.2f}",
        "Egress/Transfer Cost": f"${rpt_egress_cost:.2f}",
        "Total Estimated Cost": f"${rpt_total:.2f}"
    }
    
    if st.button("📥 Generate & Download PDF Report", type="primary", use_container_width=True):
        with st.spinner("🔄 Generating executive report..."):
            try:
                pdf_bytes = rg.generate_pdf_report(
                    platform=platform,
                    prediction=final_prediction,
                    physics_baseline=t_init_phys,
                    sla_target=rpt_sla,
                    downtime_cost=rpt_cost,
                    confidence=confidence_score,
                    model_choice=model_choice,
                    workload_config=workload_config,
                    is_breach=is_rpt_breach,
                    cost_data=cost_data
                )
                
                ts = datetime.now().strftime('%Y%m%d_%H%M')
                st.download_button(
                    label="⬇️ Download Report (.pdf)",
                    data=pdf_bytes,
                    file_name=f"GMCR_Report_{platform}_{ts}.pdf",
                    mime="application/pdf",
                    key="pdf_dl"
                )
                st.success("✅ Report generated! Click the download button above.")
            except Exception as e_rpt:
                st.error(f"❌ Report generation failed: {e_rpt}")