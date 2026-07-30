from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_PATH = Path("insurance.csv")
MODEL_PATH = Path("insurance_model.pkl")

SEX_OPTIONS = ["male", "female"]
SMOKER_OPTIONS = ["no", "yes"]
REGION_OPTIONS = ["southwest", "southeast", "northwest", "northeast"]


# --------------------------------------------------------------------------- #
# Data / model loading
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_dataset(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def build_input_frame(age, sex, bmi, children, smoker, region) -> pd.DataFrame:
    return pd.DataFrame(
        {"age": [age], "sex": [sex], "bmi": [bmi], "children": [children], "smoker": [smoker], "region": [region]}
    )


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #

def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .stApp {
            background:
                radial-gradient(circle at 15% 0%, rgba(56, 189, 248, 0.06) 0%, transparent 35%),
                radial-gradient(circle at 85% 100%, rgba(129, 140, 248, 0.06) 0%, transparent 35%),
                #0b0e14;
        }

        .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1140px; }
        #MainMenu, footer, header { visibility: hidden; }

        .card { background: #12161f; border: 1px solid #1f2531; border-radius: 18px; padding: 30px 32px; }
        .card + .card { margin-top: 18px; }

        .app-title { font-size: 32px; font-weight: 700; color: #f1f3f6; letter-spacing: -0.5px; margin-bottom: 4px; }
        .app-subtitle { font-size: 15px; color: #8b93a3; }

        .section-label {
            font-size: 12px; font-weight: 600; letter-spacing: 1.2px; text-transform: uppercase;
            color: #5b8cff; margin-bottom: 16px;
        }

        label, .stSelectbox label, .stNumberInput label { color: #b7bdc9 !important; font-weight: 500 !important; font-size: 13.5px !important; }

        .stSelectbox div[data-baseweb="select"] > div, .stNumberInput input {
            background: #191e29 !important; border: 1px solid #262d3a !important; border-radius: 10px !important; color: #f1f3f6 !important;
        }
        .stNumberInput button { background: #191e29 !important; border-color: #262d3a !important; }

        .stButton > button {
            width: 100%; height: 52px; border-radius: 10px; border: none; font-size: 15.5px; font-weight: 600;
            background: #3b82f6; color: white; transition: background 0.15s ease;
        }
        .stButton > button:hover { background: #2f6fe0; }

        .result-box {
            background: linear-gradient(135deg, #1c2b4a, #16233b); border: 1px solid #2a3f66;
            border-radius: 16px; padding: 28px 32px; margin-top: 20px;
        }
        .result-label { font-size: 13px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; color: #7fa8ff; margin-bottom: 6px; }
        .result-value { font-size: 40px; font-weight: 700; color: #f5f7fa; letter-spacing: -1px; }

        .badge {
            display: inline-block; border-radius: 999px; padding: 4px 12px; font-size: 12px; font-weight: 600; margin-right: 6px;
        }
        .badge-blue { background: rgba(59,130,246,0.15); color: #7fa8ff; border: 1px solid rgba(59,130,246,0.3); }
        .badge-green { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
        .badge-red { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }

        div[data-testid="stMetric"] { background: #12161f; border: 1px solid #1f2531; border-radius: 14px; padding: 14px; }

        .app-footer { text-align: center; margin-top: 26px; padding: 16px 0 4px 0; color: #5b6270; font-size: 12.5px; }
        .app-footer .ai-tag {
            display: inline-flex; align-items: center; gap: 6px; background: rgba(59,130,246,0.08);
            border: 1px solid rgba(59,130,246,0.18); color: #7fa8ff; border-radius: 999px; padding: 5px 14px; font-weight: 600; font-size: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    inject_css()
    model = load_model(MODEL_PATH)
    dataset = load_dataset(DATA_PATH)

    st.markdown(
        """
        <div class="card">
            <div class="app-title">💰 Medical Insurance Cost Predictor</div>
            <div class="app-subtitle">
                Estimate annual medical insurance charges using a Random Forest model,
                and see how your profile compares to the training population.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    tab_predict, tab_insights, tab_about = st.tabs(["🔮 Predict", "📊 Insights", "ℹ️ About"])

    # ------------------------------------------------------------------- #
    # Tab 1: Predict
    # ------------------------------------------------------------------- #
    with tab_predict:
        left, right = st.columns([2, 1])

        with left:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>Patient Details</div>", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Age", min_value=18, max_value=100, value=25, step=1)
                bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=26.5, step=0.1, format="%.1f")
                children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, step=1)
            with c2:
                sex = st.selectbox("Gender", SEX_OPTIONS)
                smoker = st.selectbox("Smoker", SMOKER_OPTIONS)
                region = st.selectbox("Region", REGION_OPTIONS)

            st.markdown("<br>", unsafe_allow_html=True)
            predict_clicked = st.button("Predict Insurance Cost")
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown(
                """
                <div class="card">
                    <div class="section-label">Model Info</div>
                    <p style="color:#8b93a3; font-size:14px;">
                        Random Forest Regressor trained on patient demographic
                        and lifestyle factors to estimate annual insurance charges.
                    </p>
                    <span class="badge badge-blue">Regression</span>
                    <span class="badge badge-blue">300 trees</span><br><br>
                    <span class="badge badge-blue">USD / year</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if predict_clicked:
            if model is None:
                st.error(f"Model file not found at `{MODEL_PATH}`. Run the notebook or training script first.")
            else:
                sample = build_input_frame(age, sex, bmi, children, smoker, region)
                try:
                    prediction = float(model.predict(sample)[0])
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    prediction = None

                if prediction is not None:
                    st.markdown(
                        f"""
                        <div class="result-box">
                            <div class="result-label">Estimated Annual Insurance Cost</div>
                            <div class="result-value">${prediction:,.2f}</div>
                            <div style="margin-top:10px;">
                                <span class="badge badge-blue">Age {age}</span>
                                <span class="badge badge-blue">BMI {bmi}</span>
                                <span class="badge {'badge-red' if smoker == 'yes' else 'badge-green'}">
                                    {'Smoker' if smoker == 'yes' else 'Non-smoker'}
                                </span>
                                <span class="badge badge-blue">{children} children</span>
                                <span class="badge badge-blue">{region.title()}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if dataset is not None:
                        avg_charge = dataset["charges"].mean()
                        diff_pct = (prediction - avg_charge) / avg_charge * 100
                        direction = "above" if diff_pct > 0 else "below"
                        st.caption(
                            f"This estimate is **{abs(diff_pct):.0f}% {direction}** the average charge "
                            f"(${avg_charge:,.2f}) in the training dataset."
                        )

    # ------------------------------------------------------------------- #
    # Tab 2: Insights
    # ------------------------------------------------------------------- #
    with tab_insights:
        if dataset is None:
            st.warning(f"Dataset not found at `{DATA_PATH}` — insights unavailable.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Records", len(dataset))
            m2.metric("Avg Charge", f"${dataset['charges'].mean():,.0f}")
            m3.metric("Smoker Avg", f"${dataset[dataset['smoker']=='yes']['charges'].mean():,.0f}")
            m4.metric("Non-Smoker Avg", f"${dataset[dataset['smoker']=='no']['charges'].mean():,.0f}")

            st.write("")
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<div class='section-label'>Charges by Smoking Status</div>", unsafe_allow_html=True)
                smoker_avg = dataset.groupby("smoker")["charges"].mean().rename({"no": "Non-Smoker", "yes": "Smoker"})
                st.bar_chart(smoker_avg, color="#3b82f6")
                st.markdown("</div>", unsafe_allow_html=True)

            with c2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<div class='section-label'>Average Charges by Region</div>", unsafe_allow_html=True)
                region_avg = dataset.groupby("region")["charges"].mean()
                st.bar_chart(region_avg, color="#34d399")
                st.markdown("</div>", unsafe_allow_html=True)

            st.write("")
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>Charge Distribution</div>", unsafe_allow_html=True)
            st.bar_chart(dataset["charges"].value_counts(bins=25).sort_index(), color="#c084fc")
            st.caption("Distribution of insurance charges across the training dataset.")
            st.markdown("</div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------- #
    # Tab 3: About
    # ------------------------------------------------------------------- #
    with tab_about:
        st.markdown(
            """
            <div class="card">
                <div class="section-label">About this project</div>
                <p style="color:#c7cdda; line-height:1.7;">
                    This app estimates annual medical insurance charges from six patient
                    attributes: age, sex, BMI, number of children, smoker status, and region.
                    The model is a Random Forest Regressor trained on a dataset matching the
                    schema of the widely-used Kaggle "Medical Cost Personal Datasets".
                </p>
                <div class="section-label" style="margin-top:20px;">Dataset note</div>
                <p style="color:#8b93a3; font-size:13.5px; line-height:1.7;">
                    This environment cannot fetch data directly from kaggle.com, so
                    <code>data/insurance.csv</code> is synthetically generated to match that
                    dataset's schema, ranges, and well-documented real-world cost relationships
                    (1,338 records; smoking and high BMI drive charges up sharply; age raises the
                    baseline; region adds modest variance). Swap in the original Kaggle CSV at the
                    same path if you have direct access — no other code needs to change.
                </p>
                <div class="section-label" style="margin-top:20px;">Tech stack</div>
                <span class="badge badge-blue">Streamlit</span>
                <span class="badge badge-blue">scikit-learn</span>
                <span class="badge badge-blue">Pandas</span>
                <span class="badge badge-blue">Random Forest</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
