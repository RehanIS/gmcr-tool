# 🌍 Cohesity GMCR: Google Maps for Cyber Recovery

### *Precision Orchestration & Time-to-Recovery (TTR) Prediction*

**Cohesity GMCR** is an intelligent orchestration engine designed to answer the most critical question during a cyber crisis: *"How long until we are back online?"*

Just as Google Maps predicts your arrival time (ETA) by analyzing distance, speed limits, and real-time traffic congestion, **GMCR** predicts your Data Recovery time by analyzing Data Size, Bandwidth Physics, and Cloud API "Friction."

---

## 🧭 The Core Concept

Recovery is not just `Size / Speed`. Real-world recovery involves hidden variables—API throttling, disk I/O latency, network handshake overhead, and region-specific lag.

**GMCR operates on three layers of intelligence:**
1.  **The Physics Layer (The Route):** The raw mathematical baseline (Distance / Speed Limit).
2.  **The AI Layer (The Traffic):** Machine Learning models that learn "Road Conditions" (Cloud Congestion) from historical logs.
3.  **The Simulation Layer (The Weather):** Real-time stress testing to see how volatility impacts stability.

---

## 🛠️ System Architecture

The application is built on a modular Python stack:
* **Frontend:** [Streamlit](https://streamlit.io/) for the interactive dashboard.
* **Data Processing:** `Pandas` & `NumPy` for high-speed vector manipulation.
* **AI Core:** `Scikit-Learn` (Random Forest, Gradient Boosting) for predictive modeling.
* **Visualization:** `Plotly` for interactive financial-grade charting.

---

## 🧮 1. The Physics Engine (Theoretical Formula)

When the **"Theoretical Physics"** mode is selected, the tool bypasses AI and uses a deterministic Linear Regression formula derived from network transport principles.

### The Equation
$$T_{total} = \beta_0 + (\beta_1 \cdot S) + (\beta_2 \cdot RTT) + (\beta_3 \cdot N) + (\beta_4 \cdot L) + \epsilon$$

### Variable Breakdown
| Symbol | Variable | Definition | Impact |
| :--- | :--- | :--- | :--- |
| **$T$** | **Time** | The final predicted recovery time (Minutes). | *Output* |
| **$\beta_0$** | **Intercept** | Startup Overhead. The "Handshake Time" required to authenticate and spin up resources before 1 byte is moved. | **Constant Penalty** |
| **$S$** | **Size** | Total Data Volume (GB). | **Primary Driver** |
| **$RTT$** | **Latency** | Round-Trip Time (ms). A proxy for the geographic distance between the Backup Vault and the Target VM. | **Linear Penalty** |
| **$N$** | **Concurrency** | Number of parallel threads/streams. Higher $N$ reduces time (up to a saturation point). | **Accelerator** |
| **$L$** | **Load** | Network Congestion Factor ($0.0 - 1.0$). Represents packet loss or shared bandwidth contention. | **Exponential Penalty** |

---

## 🧠 2. The AI Engines (The Brains)

The tool offers selectable algorithms to fit different data maturity levels.

### A. Gradient Boosting (The Specialist)
* **Logic:** Builds trees sequentially. Each new tree focuses *only* on the errors made by the previous tree.
* **Best For:** Complex cloud environments where "Cloud Friction" (throttling) is non-linear and hard to predict.

### B. Random Forest (The Generalist)
* **Logic:** Builds 100+ "Parallel Universes" (Trees) and averages their opinion.
* **Best For:** Noisy data sets with outliers. It is harder to fool than a single model.

### C. Linear Regression (The Baseline)
* **Logic:** Draws a straight line of best fit.
* **Best For:** proving that cloud recovery is *not* linear. (Used mostly for baseline comparison).

---

## 📊 Feature Walkthrough

### Tab 1: Planner & Predictor
This is the Command Center.
* **Input Vector:** You define the scenario (Cloud Platform, Region, Disk Type, Bandwidth).
* **Cloud Friction Metric:** The tool calculates the difference between *Physics* (Perfect World) and *AI Prediction* (Real World).
    * *Example:* `Physics: 10m` | `AI: 15m` | `Friction: +5m (33%)`.
    * This tells the SRE that 33% of their time is being lost to inefficiencies, not data size.

### Tab 2: Dynamic Simulator
Simulates the *instability* of a recovery job.
* **Why?** An "Average Speed" of 500 Mbps is misleading if it drops to 0 Mbps every 10 seconds.
* **How:** It injects Gaussian Noise (Randomness) into the transfer loop to visualize how volatility extends the recovery timeline.

### Tab 3: Validation Lab
The "Trust" mechanism.
* **Scatter Plot:** compares `Actual Historical Time` vs `Predicted Time`.
* **The "Line of Truth":** A red dashed diagonal line. If the dots hug this line, the model is accurate. If they scatter wide, the model needs more training data.

---

## 🧹 Data Refinery (Outlier Filter)

Located in the Sidebar, this tool sanitizes the training data.

* **The Problem:** Historical logs contain "Garbage Data."
    * *Example A:* A job failed instantly (0 seconds).
    * *Example B:* A job hung for 48 hours due to a power outage.
* **The Solution:** The slider sets a Quantile Range (e.g., `0.05 - 0.95`). This tells the AI: *"Ignore the bottom 5% and top 5% of extreme cases. Learn only from the normal 90%."*

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* Pip (Python Package Installer)

### Installation
1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/YourUsername/Cohesity-GMCR.git](https://github.com/YourUsername/Cohesity-GMCR.git)
    cd Cohesity-GMCR
    ```

2.  **Install Dependencies**
    ```bash
    pip install streamlit pandas numpy plotly scikit-learn
    ```

3.  **Run the Engine**
    ```bash
    streamlit run main.py
    ```

---

### 📂 Directory Structure

Cohesity-GMCR/ 
├── data/ # CSV Training Data (Must be present) 
├── assets/ # Logos and Images 
├── ai_models.py # The ML Training & Prediction Logic 
├── data_loader.py # Data Cleaning & Vectorization Logic 
├── main.py # The Streamlit Frontend Application 
├── guideme.md # You are reading this 
└── README.md # GitHub Landing Page
