# 🛡️ Cohesity-GMCR: Physics-Informed Cyber Recovery Twin

### 🚀 Overview
**GMCR (Global Management & Cloud Recovery)** is an orchestration tool designed to predict and simulate Recovery Time Objectives (RTO) for cloud workloads. 

![Status](https://img.shields.io/badge/Status-Production%20Ready-success)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)

Unlike standard calculators, this tool uses a **"Physics-Informed AI"** approach:
1.  **Physics Layer:** Calculates the theoretical speed limit (Time = Size / Bandwidth).
2.  **AI Layer:** Uses Machine Learning (Gradient Boosting) to learn "Cloud Friction" (latency, throttling, overhead) from historical data.
3.  **Result:** A highly accurate prediction that accounts for real-world chaos.

---

### 🌟 Key Features
* **Multi-Cloud Support:** Tailored models for **Azure, AWS, and VMware**.
* **Data Refinery:** Interactive slider to filter out statistical outliers (stuck jobs or logging errors) from training data.
* **Hybrid Prediction Engine:** Choose between purely theoretical formulas or AI-driven models (Random Forest, Gradient Boosting).
* **Dynamic Simulator:** Visualizes the recovery process with adjustable network volatility (noise) to simulate real-world instability.

---

### 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/Cohesity-GMCR.git](https://github.com/YOUR_USERNAME/Cohesity-GMCR.git)
    cd Cohesity-GMCR
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    streamlit run main.py
    ```

---

### 📂 Project Structure
```text
├── assets/                #logos
├── data/                  # Place your CSV datasets here
├── main.py                # The Dashboard UI (Streamlit)
├── data_loader.py         # Feature Engineering & Data Cleaning
├── ai_models.py           # ML Model Training & Validation
└── requirements.txt       # Python dependencies