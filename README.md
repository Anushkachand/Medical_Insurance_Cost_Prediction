#  Medical Insurance Cost Predictor
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://medicalinsurancecostprediction-empspbccyvix2bfu2pvrw2.streamlit.app/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)
 
**A machine learning web application that estimates annual medical insurance charges from patient profile data.**


## ✨ Features

- 🔮 Predict annual medical insurance charges
- 📊 Interactive and responsive Streamlit interface
- 🧠 Random Forest Regression model
- 📈 Dataset insights and visualizations
- 👤 Patient profile input
- ⚡ Instant prediction results
- 📱 Responsive UI
- 🎨 Clean and modern dashboard


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

# 🛠️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Joblib
- ----

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

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/Anushkachand/Medical_Insurance_Cost_Prediction.git
```

Move into the project

```bash
cd Medical_Insurance_Cost_Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 🌐 Deployment

This project is deployed on **Streamlit Community Cloud**.

### Live Application

https://medicalinsurancecostprediction-empspbccyvix2bfu2pvrw2.streamlit.app/

---
The app has three tabs:
- **🔮 Predict** — enter patient details, get an instant cost estimate
- **📊 Insights** — see how the dataset breaks down by smoking status, region, and overall charge distribution, plus how your prediction compares to the dataset average
- **ℹ️ About** — project, dataset, and model details

## ⚠️ Disclaimer

This is an educational / portfolio project. Predictions are estimates from a
model trained on synthetic data and are **not financial or medical advice**.

---
 
## Contact
 
**Anushka Chand**
 
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/anushka-chand-18ab44300/)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?logo=gmail&logoColor=white)](mailto:anushkakaushik0801@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-Anushkachand-181717?logo=github&logoColor=white)](https://github.com/Anushkachand)
---
