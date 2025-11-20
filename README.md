# Bankruptcy Prevention

> **Predicting business bankruptcy using interpretable machine learning**

---

## 🚀 Project Overview

**Bankruptcy Prevention** is a binary classification project that models the probability a company will go bankrupt using a small, high-impact dataset. The dataset contains ~250 companies and 7 curated features capturing industrial, managerial, financial, and operational signals. The goal is to build a robust, explainable pipeline that flags at-risk companies early so stakeholders can intervene.

---

## 🎯 Business Objective

This is a classification problem (bankruptcy vs non-bankruptcy). The objective is to model and estimate the probability that a business becomes bankrupt based on several risk and capability indicators, so that risk teams can prioritize actions and prevent failures.

---

## 📦 Dataset (features)

The dataset includes the following variables (each scaled discretely as 0, 0.5, 1):

* **industrial_risk** — 0 = low risk, 0.5 = medium risk, 1 = high risk
* **management_risk** — 0 = low risk, 0.5 = medium risk, 1 = high risk
* **financial_flexibility** — 0 = low flexibility, 0.5 = medium flexibility, 1 = high flexibility
* **credibility** — 0 = low credibility, 0.5 = medium credibility, 1 = high credibility
* **competitiveness** — 0 = low competitiveness, 0.5 = medium competitiveness, 1 = high competitiveness
* **operating_risk** — 0 = low risk, 0.5 = medium risk, 1 = high risk

> Dataset size: ~250 company records, 7 features + target (bankruptcy flag).

---

## 🧭 What the notebook contains

See the included Jupyter notebook for the code and step-by-step work:

**Notebook path:** `/mnt/data/P596_Bankruptacy_Prevention.ipynb`

The notebook implements the following stages:

1. **Data loading & quick sanity checks** — header, missing values, basic statistics
2. **Exploratory Data Analysis (EDA)** — distribution plots, correlation matrix, and risk profiling
3. **Preprocessing & Feature Engineering** — scaling, encoding (if needed), train-test split
4. **Modeling** — baseline models (Logistic Regression, Decision Tree), and one or more advanced models (RandomForest, XGBoost or LightGBM) with hyperparameter tuning
5. **Evaluation** — confusion matrix, ROC-AUC, precision/recall, classwise metrics
6. **Explainability** — feature importances and SHAP/permute-based explanations to show drivers of bankruptcy predictions
7. **Saving & reproducibility** — model export (joblib/pickle), reproducible seeds, and short notes on deployment

---

## 🛠 How to run the project (local)

1. Clone this repo or copy files into a working directory.
2. Open the notebook at: `/mnt/data/P596_Bankruptacy_Prevention.ipynb`.
3. Create a virtual environment and install requirements (example):

```bash
python -m venv venv
source venv/bin/activate   # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

4. Start Jupyter and run the notebook cells:

```bash
jupyter notebook /mnt/data/P596_Bankruptacy_Prevention.ipynb
```

> Tip: If you want to run end-to-end faster, execute the cells that load pre-saved model artifacts (if included) and skip heavy hyperparameter search sections.

---

## 📈 Results & Outputs

* **Best model:** `Logistic Regression`
* **Top features driving bankruptcy:**
  ` 1. Management Risk  
    2. Financial Flexibility
    3. Credibility
    4. Competitiveness `

---

## 🖼 Output image

<img width="1912" height="932" alt="b" src="https://github.com/user-attachments/assets/86d5b0df-1af4-4488-9262-e1825197adfe" />

<img width="1912" height="932" alt="n" src="https://github.com/user-attachments/assets/ed222117-04ea-4186-91b7-a4ae45733efa" />

---

## ✅ Key takeaways (for stakeholders)

* The model provides a **probabilistic risk score** so teams can prioritize companies by risk tier rather than binary yes/no.
* Explainability (SHAP or feature importances) highlights which factors contribute most to default risk — enabling targeted interventions (management coaching, liquidity support, etc.).
* With a small, well-engineered feature set, this system is lightweight, fast, and easy to integrate into monitoring dashboards.

---

## 📦 Requirements (suggested)

* Python 3.9+
* pandas, numpy, scikit-learn
* matplotlib, seaborn, plotly (optional for visualization)
* joblib or pickle
* shap (optional but recommended for XAI)
* xgboost or lightgbm (optional if used)

---

## 🧩 Future work & improvements

* Expand the dataset with longitudinal financials and macro indicators.
* Add time-series features (trend of key metrics) and survival analysis for time-to-bankruptcy predictions.
* Implement a lightweight API for real-time scoring and a monitoring pipeline (MLflow / Prometheus + Grafana).

---

## 📝 License & Contact

This project is open for iteration and learning.

---

*Created with ❤️ — edit the placeholders above (metrics, model name, image path) after you run the notebook once to capture final outputs.*
