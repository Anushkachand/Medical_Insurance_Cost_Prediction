# 💰 Medical Insurance Cost Predictor

A Streamlit app that estimates annual medical insurance charges from patient
profile data (age, sex, BMI, children, smoker status, region) using a trained
Random Forest regression model.

## 📁 Project Structure

```
medical_insurance_predictor/
├── app.py                     → Streamlit UI (Predict / Insights / About tabs)
├── requirements.txt           → Python dependencies
├── medical.ipynb              → EDA + model training notebook
├── README.md                  → this file
├── data/
│   ├── generate_dataset.py    → builds insurance.csv
│   └── insurance.csv          → training dataset (1,338 records)
└── model/
    └── insurance_model.pkl    → trained Random Forest model
```

## 📊 Dataset

**Schema:** `age, sex, bmi, children, smoker, region, charges` — matching the
widely-used Kaggle **"Medical Cost Personal Datasets"** (by Miri Choi), 1,338
records.

**Important note:** this sandboxed build environment cannot reach
`kaggle.com` directly, so `data/insurance.csv` here is **synthetically
generated** (`data/generate_dataset.py`) to match that dataset's exact column
schema, value ranges, and well-documented real-world cost relationships:

- Smoking status has the strongest effect on charges (sharp increase)
- Higher BMI compounds cost, especially combined with smoking
- Age raises the baseline charge steadily
- Each dependent (child) adds a modest premium
- Region contributes small variance

If you have direct access to the original Kaggle CSV, you can drop it in at
`data/insurance.csv` — the schema is identical, so no other code needs to
change.

## 🧠 Model

- **Algorithm:** Random Forest Regressor (300 trees, max depth 8)
- **Preprocessing:** One-hot encoding for `sex`, `smoker`, `region`
- **Target:** `charges` (USD/year)
- **Performance (test split):** R² ≈ 0.98, MAE ≈ $1,300 (see `medical.ipynb` for exact numbers and plots)

Retrain anytime with:
```bash
jupyter nbconvert --to notebook --execute medical.ipynb
```
or open `medical.ipynb` directly in Jupyter/VS Code and run all cells — it
regenerates and overwrites `model/insurance_model.pkl`.

## 🖥️ Running the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app has three tabs:
- **🔮 Predict** — enter patient details, get an instant cost estimate
- **📊 Insights** — see how the dataset breaks down by smoking status, region, and overall charge distribution, plus how your prediction compares to the dataset average
- **ℹ️ About** — project, dataset, and model details

## ⚠️ Disclaimer

This is an educational / portfolio project. Predictions are estimates from a
model trained on synthetic data and are **not financial or medical advice**.

---
✨ Built with AI (Claude).
