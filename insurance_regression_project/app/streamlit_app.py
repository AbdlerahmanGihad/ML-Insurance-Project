import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

st.set_page_config(page_title="Medical Insurance Cost Prediction", page_icon="🏥", layout="centered")

st.markdown("""
<style>
.stApp{
    background-color:#ffffff;
}

h1,h2,h3{
    color:#111827;
}

p{
    color:#374151;
}

/* Buttons */
.stButton > button{
    background-color:#1B5E20 !important;
    color:#FFFFFF !important;
    border:none !important;
    border-radius:8px !important;
    padding:0.6rem 1.5rem !important;
    font-weight:600 !important;
}

.stButton > button:hover{
    background-color:#2E7D32 !important;
    color:#FFFFFF !important;
}

.stButton > button p,
.stButton > button span{
    color:#FFFFFF !important;
}

/* Result Card */
.result-card{
    background:#f5f5f5;
    padding:20px;
    border-radius:10px;
}

.result-value{
    color:#1B5E20;
    font-size:36px;
    font-weight:bold;
}

.result-card div{
    color:#111827 !important;
}

.result-value{
    color:#1B5E20 !important;
    font-size:36px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

MODELS = {
    "Random Forest": {
        "file": MODELS_DIR / "random_forest.pkl",
        "r2": 0.901,
        "mae": 2415,
        "rmse": 4262,
    },
    "Decision Tree": {
        "file": MODELS_DIR / "decision_tree.pkl",
        "r2": 0.892,
        "mae": 2693,
        "rmse": 4461,
    },
    "Linear Regression": {
        "file": MODELS_DIR / "linear_regression.pkl",
        "r2": 0.886,
        "mae": 2829,
        "rmse": 4573,
    }
}

FEATURE_COLUMNS = joblib.load(MODELS_DIR / "feature_columns.pkl")
SCALER = joblib.load(MODELS_DIR / "scaler.pkl")


def build_features(age, sex, bmi, children, smoker, region):
    row = {col: 0 for col in FEATURE_COLUMNS}
    row["age"] = age
    row["sex"] = 1 if sex == "Male" else 0
    row["bmi"] = bmi
    row["children"] = children
    row["smoker"] = 1 if smoker == "Yes" else 0
    row["bmi_smoker"] = bmi * row["smoker"]

    region_col = f"region_{region.lower()}"
    if region_col in row:
        row[region_col] = 1

    return pd.DataFrame([row])[FEATURE_COLUMNS]


st.title("🏥 Medical Insurance Cost Prediction")
st.write("Predict medical insurance charges using Machine Learning.")

st.markdown("""
This app estimates a person's yearly medical insurance cost based on their age, gender,
BMI, number of children, smoking status, and region. Three regression models were trained
on 1,338 real insurance records and compared, and you can pick which one to use below.
The prediction is generated instantly on your input, nothing is stored.
""")

st.subheader("Patient Information")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    children = st.number_input("Children", min_value=0, max_value=10, value=0)
with col2:
    sex = st.selectbox("Gender", ["Male", "Female"])
    smoker = st.selectbox("Smoker", ["No", "Yes"])
    region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

st.subheader("Model")
model_name = st.selectbox("Choose a model", list(MODELS.keys()), index=0)

if st.button("Predict"):
    model = joblib.load(MODELS[model_name]["file"])
    features = build_features(age, sex, bmi, children, smoker, region)

    if model_name == "Linear Regression":
        numerical_cols = ["age", "bmi", "children", "bmi_smoker"]
        features[numerical_cols] = SCALER.transform(features[numerical_cols])

    prediction = model.predict(features)[0]

    st.markdown(f"""
    <div class="result-card">
        <div>Predicted Insurance Cost</div>
        <div class="result-value">${prediction:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Model Performance")
    info = MODELS[model_name]
    m1, m2, m3 = st.columns(3)
    m1.metric("R2 Score", f"{info['r2']:.2f}")
    m2.metric("MAE", f"{info['mae']:,}")
    m3.metric("RMSE", f"{info['rmse']:,}")

st.divider()

if st.button("Show Model Comparison"):
    comparison = pd.DataFrame([
        {"Model": name, "R2": v["r2"], "MAE": v["mae"], "RMSE": v["rmse"]}
        for name, v in MODELS.items()
    ])
    st.dataframe(comparison, hide_index=True, use_container_width=True)
    st.caption("Random Forest has the best R2 and lowest error, so it is the default model above.")

with st.expander("How does this work?"):
    st.markdown("""
    1. The data was cleaned and encoded (`1_preprocessing.ipynb`).
    2. Three regression models were trained and tuned (`3_modeling.ipynb`):
       Linear Regression, Decision Tree, and Random Forest.
    3. Models were compared using R2, MAE, and RMSE on unseen test data (`4_evaluation.ipynb`).
    4. This app loads the saved models and applies the same preprocessing steps
       to your input, then returns a prediction from the model you choose.

    **To run this app locally:**
    ```
    cd app
    streamlit run streamlit_app.py
    ```
    """)