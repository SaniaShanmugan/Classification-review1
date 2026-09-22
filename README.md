# 🏦 Loan Approval Classification — ML Capstone (Review 1)
**Course:** 23CSE301 Machine Learning | **Track:** Classification (Part A) | **Review:** 1  

An end-to-end Machine Learning pipeline and interactive web application for predicting retail loan approvals based on applicant demographics, requested loan parameters, credit bureau track records (CIBIL score), and declared asset collateral.

---

## 📌 Project Overview
Predicting loan approval is a critical credit underwriting task for financial institutions. This project implements, evaluates, and compares five foundational classification algorithms on **4,269 real loan applications**:
1. **Decision Tree Classifier** (tuned `max_depth=4`)
2. **Support Vector Machine (SVC)** (tuned $C=5$, RBF kernel)
3. **Logistic Regression** (regularized $C=1.0$)
4. **K-Nearest Neighbors (KNN)** (tuned $k=21$, Euclidean)
5. **Gaussian Naive Bayes** (probabilistic baseline)

All scalers and encoders are strictly fitted on the training split only (`X_train`) to guarantee **zero data leakage**, and top models are validated using **5-Fold Stratified Cross-Validation**.

---

## 🏆 Model Performance Leaderboard (Held-Out 20% Test Set)

| Rank | Model | Accuracy | Precision (Weighted) | Recall (Weighted) | Weighted F1 | ROC-AUC | TN | FP | FN | TP | Fit Time |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Decision Tree (`depth=4`)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 323 | 0 | 0 | 531 | 0.008s |
| **2** | **Support Vector Machine (SVC)** | **0.9485** | **0.9484** | **0.9485** | **0.9483** | **0.9918** | 297 | 26 | 18 | 513 | 0.015s |
| **3** | **Logistic Regression** | **0.9251** | **0.9249** | **0.9251** | **0.9249** | **0.9790** | 287 | 36 | 28 | 503 | 0.016s |
| **4** | **KNN (`k=21`)** | **0.9075** | **0.9083** | **0.9075** | **0.9064** | **0.9738** | 267 | 56 | 23 | 508 | 0.003s |
| **5** | **Gaussian Naive Bayes** | **0.7518** | **0.7609** | **0.7518** | **0.7327** | **0.7310** | 149 | 174 | 38 | 493 | 0.004s |

### 5-Fold Stratified Cross-Validation (Top-2 Models)
Scaling embedded in a scikit-learn `Pipeline` to prevent cross-fold leakage:
- **Decision Tree (`depth=4`)**: Mean Accuracy = `1.0000 (±0.0000)` | Mean Weighted F1 = `1.0000 (±0.0000)` | Mean ROC-AUC = `1.0000 (±0.0000)`
- **SVC ($C=5$, RBF)**: Mean Accuracy = `0.9467 (±0.0144)` | Mean Weighted F1 = `0.9468 (±0.0142)` | Mean ROC-AUC = `0.9870 (±0.0037)`

---

## 🗂️ Dataset & Engineered Features
- **Dataset:** `loan_approval_dataset.csv` (4,269 records, 13 raw attributes, 0 missing values)
- **Target:** `loan_status` (Approved: 62.2% | Rejected: 37.8%)
- **Engineered Financial Indicators (§3.4):**
  1. `Total_Assets`: Sum of residential, commercial, luxury, and bank liquid assets.
  2. `Loan_to_Income_Ratio`: Leverage ratio (`loan_amount / income_annum`).
  3. `Asset_to_Loan_Ratio`: Collateral coverage multiplier (`Total_Assets / loan_amount`).
  4. `Estimated_EMI`: Monthly repayment cashflow burden assuming standard 8% p.a. over `loan_term` years.
  5. `Income_per_Dependent`: Household per-capita disposable income.

---

## 📁 Repository Structure
```
├── classification_final.ipynb   # Polished, fully run classification notebook with all outputs
├── app.py                       # Interactive Streamlit web application
├── loan_approval_dataset.csv    # Loan approval dataset
├── requirements.txt             # Pinned project dependencies
├── .gitignore                   # Configured gitignore for Python, Jupyter & Streamlit
└── README.md                    # Project overview & documentation
```

---

## 🚀 Setup & Execution Guide

### 1. Clone & Environment Setup
```bash
git clone <your-repository-url>
cd Sania
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the Jupyter Notebook
Open `classification_final.ipynb` in VS Code or JupyterLab:
```bash
jupyter lab
```
All 56 cells are already executed top-to-bottom with full visual outputs and tables visible.

### 3. Launch the Streamlit Web Application
Run the interactive application:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser. The app features:
- **🏠 Overview:** Project context and dataset schema.
- **📊 EDA:** Interactive distribution plots, correlation matrix, and CIBIL decision boundary scatter plot.
- **🏆 Model Comparison:** Leaderboard and confusion matrices.
- **🔍 Model Deep Dive:** Decision tree visualizer, odds ratios, and ROC curves.
- **🧪 Predict with Your Input:** Real-time applicant scoring panel with instant consensus predictions across all 5 models.
