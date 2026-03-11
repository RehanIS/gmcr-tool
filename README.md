# 🛡️ GMCR: Google Maps for Cyber Recovery

### *Predicting the "ETA" of Enterprise Data Recovery with ML*

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B)
![AI](https://img.shields.io/badge/AI-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 📖 The Concept
**When disaster strikes, the only question that matters is: *"How long until we are back?"***

Legacy tools give you a simple math calculation: `Size / Speed = Time`.
**GMCR (Google Maps for Cyber Recovery)** goes deeper. It acts as the "Google Maps" for your recovery journey. Just as Google Maps accounts for traffic, accidents, and road quality—not just distance—GMCR accounts for **API Throttling, Cloud Friction, Network Jitter, and Regional Latency.**

It replaces "Guesstimates" with **Data Science.**

---

## ⚡ Key Features

### 1. 🧠 Multi-Engine Intelligence
Choose the "Brain" that fits your scenario:
* **Gradient Boosting:** The "Expert" mode. Detects complex non-linear patterns like cloud throttling.
* **Random Forest:** The "Robust" mode. Best for noisy, inconsistent network logs.
* **Theoretical Physics:** The "Baseline" mode. Uses a raw linear regression formula (see below) for deterministic planning.

### 2. 🧮 The "Physics of Recovery" Formula
When AI is disabled, the tool uses a custom Linear Regression model to calculate the theoretical best-case scenario:

$$T_{total} = \beta_0 + (\beta_1 \cdot S) + (\beta_2 \cdot RTT) + (\beta_3 \cdot N) + (\beta_4 \cdot L)$$

| Var | Definition | Real-World Meaning |
| :--- | :--- | :--- |
| **$S$** | **Size** | The volume of data (GB) being hydrated. |
| **$RTT$** | **Latency** | Distance penalty based on Cloud Region (e.g., East US vs. Asia). |
| **$N$** | **Concurrency** | Parallel streams. More isn't always faster (diminishing returns). |
| **$L$** | **Load** | Network congestion factor mimicking "Rush Hour" traffic. |

### 3. 🕹️ Dynamic Volatility Simulator
Don't just see the *End Time*—visualize the *Journey*. The built-in simulator injects random noise (packet loss/jitter) into a transfer bar to show stakeholders why "Average Speed" is a dangerous metric.

### 4. 🧹 Data Refinery
A built-in ETL (Extract, Transform, Load) slider that allows SREs to filter out statistical outliers—removing "Garbage Data" (e.g., 0-second restores or multi-day outages) from the training set.

---

## 🚀 Quick Start

### Prerequisites
* Python 3.8 or higher.
* Pip package manager.

### Installation

1.  **Clone the Repo**
    ```bash
>     git clone [https://github.com/YourUsername/gmcr-beta.git](https://github.com/YourUsername/gmcr-beta.git)
    cd gmcr-beta
    ```

2.  **Install Dependencies**
    ```bash
    pip install requirements.txt
    ```

3.  **Launch the App**
    ```bash
    streamlit run main.py
    ```
    *The dashboard will automatically open in your default browser at `http://localhost:8501`.*

---

## 📂 Project Structure

```bash
gmcr-beta/
├── data/                 # 🗄️ Historical CSV Logs (Azure/AWS/VMware)
├── assets/               # 🖼️ UI Assets & Logos
├── ai_models.py          # 🧠 The Machine Learning Logic (Training/Prediction)
├── data_loader.py        # 🧹 Data Cleaning & Vectorization Pipeline
├── main.py               # 🖥️ The Streamlit Dashboard (Frontend)
├── guideme.md            # 📘 Detailed User Manual & Physics Explanation
└── README.md             # 🏠 You are here
