# ============================================================
# Bankruptcy Predictor — Model Deployment using Streamlit
# ============================================================

import pandas as pd
import numpy as np
import streamlit as st
import joblib
from fpdf import FPDF
import plotly.graph_objects as go

# ------------------------------------------------------------
# Page Config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Bankruptcy Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# Custom Styling
# ------------------------------------------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #e3f2fd 0%, #f5f7fa 100%);
    color: #2C3E50;
    font-family: "Poppins", sans-serif;
}
h1, h2, h3 {
    color: #1A5276;
    font-weight: 700;
}
.styled-card {
    background-color: #ffffff;
    padding: 1.2em;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 1.5rem;
}
.stButton>button {
    background-color: #2E86C1;
    color: white;
    font-weight: 600;
    border-radius: 10px;
    padding: 0.6em 1.4em;
    transition: all 0.3s ease-in-out;
}
.stButton>button:hover {
    background-color: #154360;
    transform: scale(1.05);
}
.stProgress > div > div > div > div {
    background-color: #E74C3C;
}
footer {visibility: hidden;}
section[data-testid="stSidebar"], .block-container, div[data-testid="stVerticalBlock"] {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------
@st.cache_data
def load_model(path="bankruptcy_model.joblib"):
    return joblib.load(path)

@st.cache_data
def load_data(path="Bankruptcy.xlsx"):
    try:
        return pd.read_excel(path)
    except Exception:
        return pd.read_excel(path, sheet_name=0)

def build_input_ui(features, df, prefill=None):
    st.subheader("🧾 Input Financial Indicators")
    inputs = {}
    cols = st.columns(3)
    for i, feat in enumerate(features):
        col = cols[i % 3]
        # Prefer value from uploaded CSV
        if prefill and feat in prefill:
            try:
                default = float(prefill[feat])
            except ValueError:
                default = float(df[feat].median())
        else:
            default = float(df[feat].median()) if feat in df.columns else 0.0

        # Define input bounds
        min_val = float(df[feat].min()) if feat in df.columns else 0.0
        max_val = float(df[feat].max()) if feat in df.columns else default * 10 + 1
        step = (max_val - min_val) / 100 if max_val != min_val else 1.0
        label = feat.replace("_", " ").title()
        inputs[feat] = col.number_input(
            label,
            value=default,
            min_value=min_val,
            max_value=max_val,
            step=step,
            key=feat  # prevents Streamlit rerun confusion
        )
    return inputs


def generate_pdf(title, inputs, predicted_class, probability):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, title, ln=True, align='C')
    pdf.ln(8)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, f"Predicted Class: {predicted_class}", ln=True)
    pdf.cell(0, 8, f"Bankruptcy Probability: {probability*100:.2f}%", ln=True)
    pdf.ln(6)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "Input Features:", ln=True)
    pdf.set_font("Arial", size=10)
    for k, v in inputs.items():
        pdf.multi_cell(0, 6, f"- {k}: {v}")
    return pdf.output(dest='S').encode('latin-1')

def safe_predict(model, inputs, features):
    X_df = pd.DataFrame([inputs], columns=features)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_df)[0]
        prob_bankrupt = probs[0]
    else:
        prob_bankrupt = float(model.predict(X_df)[0])
    pred_class = model.predict(X_df)[0]
    return pred_class, prob_bankrupt

# ------------------------------------------------------------
# Load Model & Data
# ------------------------------------------------------------
with st.spinner("🔄 Loading model and data..."):
    model = load_model()
    df = load_data()

# Fixed feature set and target
features = [
    "industrial_risk",
    "management_risk",
    "financial_flexibility",
    "credibility",
    "competitiveness",
    "operating_risk"
]
target_col = "class"

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.title("🧭 Menu")
page = st.sidebar.radio(" ", ["🏠 Home", "📊 Predict", "ℹ️ About"])
st.sidebar.markdown("---")
st.sidebar.caption("Built using Streamlit")

# ------------------------------------------------------------
# HOME PAGE
# ------------------------------------------------------------
if page == "🏠 Home":
    st.title("Bankruptcy Predictor")
    st.markdown("### A Smart, Data-Driven Risk Assessment System")

    left, right = st.columns([2, 1])
    with left:
        with st.container():
            #st.markdown('<div class="styled-card">', unsafe_allow_html=True)
            st.success("💡 Assess bankruptcy risk based on financial indicators.")
            st.write("""
            This app loads a **pre-trained Machine Learning model** to predict bankruptcy probability.  
            You can enter indicators manually or upload a CSV file.  
            It displays both classification and probability and allows PDF report download.
            """)
            st.info(f"✅ Detected **{len(features)}** financial indicators from the dataset.")
            st.dataframe(df.head(5))
            #st.markdown('</div>', unsafe_allow_html=True)

    with right:
        med_input = pd.DataFrame([df[features].median()], columns=features)
        cls, med_prob = safe_predict(model, dict(med_input.iloc[0]), features)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=med_prob * 100,
            title={'text': "Median-case Bankruptcy %"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "crimson"}}
        ))
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# PREDICT PAGE
# ------------------------------------------------------------
elif page == "📊 Predict":
    st.title("📈 Predict Bankruptcy Risk")
    with st.container():
        #st.markdown('<div class="styled-card">', unsafe_allow_html=True)

        with st.expander("📤 Upload CSV to prefill inputs"):
            uploaded = st.file_uploader("Upload a CSV (1 row only)", type=['csv'])
            prefill = None
            if uploaded is not None:
                uploaded_df = pd.read_csv(uploaded)
                if uploaded_df.shape[0] > 1:
                    st.warning("⚠️ Only the first row will be used.")
                prefill = uploaded_df.iloc[0].to_dict()

        inputs = build_input_ui(features, df, prefill=prefill)


        if st.button("🔍 Predict"):
            pred_class, prob = safe_predict(model, inputs, features)
            st.markdown("## 🧮 Prediction Result")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Class", str(pred_class))
                st.progress(min(max(prob, 0.0), 1.0))

            with col2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    title={'text': "Bankruptcy Probability (%)"},
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "crimson"}}
                ))
                st.plotly_chart(fig, use_container_width=True)

            if prob >= 0.66:
                st.error(f"🚨 High risk of bankruptcy — {prob*100:.2f}%")
            elif prob >= 0.33:
                st.warning(f"⚠️ Moderate risk — {prob*100:.2f}%")
            else:
                st.success(f"✅ Low risk — {prob*100:.2f}%")

            pdf_bytes = generate_pdf("Bankruptcy Prediction Report", inputs, pred_class, prob)
            st.download_button("📥 Download PDF Report", pdf_bytes,
                               file_name="bankruptcy_report.pdf", mime='application/pdf')

        #st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# ABOUT PAGE
# ------------------------------------------------------------
elif page == "ℹ️ About":
    st.title("ℹ️ About this App")
    st.markdown("""
    This app uses a trained **Machine Learning model** to predict bankruptcy risk.  
    It’s designed for financial analysts and business decision-makers.
    """)

    with st.container():
        #st.markdown('<div class="styled-card">', unsafe_allow_html=True)
        st.markdown("#### 📘 Model Information")

        # Detect actual algorithm
        try:
            if hasattr(model, "steps"):
                inner_model = model.steps[-1][1]
                inner_model_name = type(inner_model).__name__
            elif hasattr(model, "named_steps"):
                inner_model_name = type(model.named_steps.get("model", model)).__name__
            else:
                inner_model_name = type(model).__name__
        except Exception:
            inner_model_name = type(model).__name__

        st.write(f"**Model Type:** {inner_model_name}")
        st.write(f"**Target Column:** `{target_col}`")

        # Label info
        try:
            labels = model.classes_
            if len(labels) == 2:
                st.write(f"**Label Classes:** `{labels[0]}` = Bankrupt, `{labels[1]}` = Non-Bankrupt")
            else:
                st.write(f"**Label Classes:** {list(labels)}")
        except Exception:
            st.info("ℹ️ Label classes could not be automatically detected.")

        st.write(f"**Using {len(features)} financial indicators:**")
        st.write(features)
        #st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("© 2025 Bankruptcy Predictor | Built with Streamlit")
