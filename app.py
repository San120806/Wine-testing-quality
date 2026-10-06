import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Page Configuration & Modern Theme Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Wine Quality AI | ML Classification Studio",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Rich dark wine theme & glassmorphic cards)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
    }

    .main-header {
        background: linear-gradient(135deg, #4a0e17 0%, #1f080c 50%, #12141a 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        color: #ffffff;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }
    
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #ff8a9e, #ffffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .subtitle {
        color: #d1b8be;
        font-size: 1.05rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }

    .metric-badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        font-size: 0.85rem;
        color: #ffccd5;
        margin-right: 0.5rem;
        margin-top: 0.8rem;
    }

    .prediction-card {
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.25);
    }

    .pred-high {
        background: linear-gradient(135deg, rgba(22, 101, 52, 0.25) 0%, rgba(20, 83, 45, 0.12) 100%);
        border-color: #22c55e;
    }

    .pred-medium {
        background: linear-gradient(135deg, rgba(180, 83, 9, 0.25) 0%, rgba(146, 64, 14, 0.12) 100%);
        border-color: #f59e0b;
    }

    .pred-low {
        background: linear-gradient(135deg, rgba(153, 27, 27, 0.25) 0%, rgba(127, 29, 29, 0.12) 100%);
        border-color: #ef4444;
    }

    .prob-bar-container {
        margin: 1rem 0;
        background: rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .group-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f472b6;
        margin-bottom: 0.8rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Load Artifacts
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")

    model_path = os.path.join(models_dir, "model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    features_path = os.path.join(models_dir, "feature_names.pkl")
    bounds_path = os.path.join(models_dir, "outlier_bounds.pkl")
    meta_path = os.path.join(models_dir, "model_metadata.json")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)
    outlier_bounds = joblib.load(bounds_path)

    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)

    return model, scaler, feature_names, outlier_bounds, metadata

try:
    model, scaler, feature_names, outlier_bounds, metadata = load_model_artifacts()
except Exception as e:
    st.error(f"Error loading trained model artifacts: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# Preset Configurations for Testing
# -----------------------------------------------------------------------------
PRESETS = {
    "Custom (Select Your Own Values)": None,
    "Sample High Quality Wine (Score 8)": {
        "fixed acidity": 12.6,
        "volatile acidity": 0.31,
        "citric acid": 0.72,
        "residual sugar": 2.2,
        "chlorides": 0.072,
        "free sulfur dioxide": 6.0,
        "total sulfur dioxide": 29.0,
        "density": 0.9987,
        "pH": 2.88,
        "sulphates": 0.82,
        "alcohol": 12.8
    },
    "Sample Medium Quality Wine (Score 6)": {
        "fixed acidity": 8.0,
        "volatile acidity": 0.45,
        "citric acid": 0.28,
        "residual sugar": 2.1,
        "chlorides": 0.078,
        "free sulfur dioxide": 15.0,
        "total sulfur dioxide": 38.0,
        "density": 0.9965,
        "pH": 3.30,
        "sulphates": 0.65,
        "alcohol": 10.5
    },
    "Sample Low Quality Wine (Score 3-5)": {
        "fixed acidity": 7.4,
        "volatile acidity": 0.75,
        "citric acid": 0.05,
        "residual sugar": 2.5,
        "chlorides": 0.095,
        "free sulfur dioxide": 22.0,
        "total sulfur dioxide": 85.0,
        "density": 0.9978,
        "pH": 3.45,
        "sulphates": 0.48,
        "alcohol": 9.2
    }
}

# -----------------------------------------------------------------------------
# Sidebar: Controls & Model Architecture Specs
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?w=500&auto=format&fit=crop&q=80", use_container_width=True)
    st.title("Control Panel")
    
    selected_preset = st.selectbox(
        "Load Quick Preset:",
        options=list(PRESETS.keys()),
        index=0
    )
    
    st.markdown("---")
    st.subheader("Model Specs")
    st.markdown("""
    - **Architecture:** Random Forest (300 Trees)
    - **Objective:** 3-Class Categorization
      - **Low:** Quality Score $\le 5$
      - **Medium:** Quality Score $= 6$
      - **High:** Quality Score $\ge 7$
    - **Cross-Validation:** 5-Fold Stratified
    - **CV Macro F1:** **0.6060 ± 0.0432**
    - **Test Macro F1:** **0.5899**
    - **Imbalance Handling:** Balanced Class Weights
    """)

    st.markdown("---")
    st.caption("🍷 Viniculture ML Case Study • Antigravity 2.0")

# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1 class="main-title">Red Wine Quality Classifier</h1>
    <p class="subtitle">Predicting artisanal wine quality tiers from physico-chemical profiles using an ensemble Random Forest</p>
    <div>
        <span class="metric-badge">🏆 Selected Model: Random Forest</span>
        <span class="metric-badge">📊 5-Fold CV Macro F1: 0.6060</span>
        <span class="metric-badge">⚖️ Balanced Weights</span>
        <span class="metric-badge">🧪 11 Chemical Features</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Interactive Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔮 Real-Time Prediction", "🔬 Feature Importance & Insights", "📈 Benchmark & Methodology"])

# -----------------------------------------------------------------------------
# TAB 1: Real-Time Prediction
# -----------------------------------------------------------------------------
with tab1:
    preset_data = PRESETS[selected_preset]
    stats = metadata.get("feature_stats", {})

    st.markdown("### Input Physico-Chemical Parameters")
    st.caption("Adjust the sliders or select a preset from the sidebar to inspect predictions.")

    col1, col2, col3 = st.columns(3)

    input_values = {}

    with col1:
        st.markdown('<div class="group-header">1. Acidity & pH Balance</div>', unsafe_allow_html=True)
        
        input_values["fixed acidity"] = st.slider(
            "Fixed Acidity (g/dm³)",
            min_value=4.0, max_value=16.0,
            value=float(preset_data["fixed acidity"]) if preset_data else 8.3,
            step=0.1,
            help="Tartaric acid concentration responsible for wine freshness."
        )

        input_values["volatile acidity"] = st.slider(
            "Volatile Acidity (g/dm³)",
            min_value=0.10, max_value=1.60,
            value=float(preset_data["volatile acidity"]) if preset_data else 0.52,
            step=0.01,
            help="Acetic acid content; high levels cause an unpleasant vinegar taste."
        )

        input_values["citric acid"] = st.slider(
            "Citric Acid (g/dm³)",
            min_value=0.00, max_value=1.00,
            value=float(preset_data["citric acid"]) if preset_data else 0.27,
            step=0.01,
            help="Adds crispness and fruity flavor in small quantities."
        )

        input_values["pH"] = st.slider(
            "pH (Acidity level)",
            min_value=2.70, max_value=4.00,
            value=float(preset_data["pH"]) if preset_data else 3.31,
            step=0.01,
            help="Measures wine acidity (most wines are between 3.0 and 3.5)."
        )

    with col2:
        st.markdown('<div class="group-header">2. Preservation & Sulfites</div>', unsafe_allow_html=True)
        
        input_values["free sulfur dioxide"] = st.slider(
            "Free SO₂ (mg/dm³)",
            min_value=1.0, max_value=72.0,
            value=float(preset_data["free sulfur dioxide"]) if preset_data else 15.0,
            step=1.0,
            help="Active antimicrobial and antioxidant form of SO₂."
        )

        input_values["total sulfur dioxide"] = st.slider(
            "Total SO₂ (mg/dm³)",
            min_value=6.0, max_value=289.0,
            value=float(preset_data["total sulfur dioxide"]) if preset_data else 46.0,
            step=1.0,
            help="Total bound and free SO₂; high values cause pungent odor."
        )

        input_values["sulphates"] = st.slider(
            "Sulphates (potassium sulphate g/dm³)",
            min_value=0.30, max_value=2.00,
            value=float(preset_data["sulphates"]) if preset_data else 0.65,
            step=0.01,
            help="Wine additive that promotes antimicrobial action and preserves freshness."
        )

    with col3:
        st.markdown('<div class="group-header">3. Body, Sugar & Alcohol</div>', unsafe_allow_html=True)
        
        input_values["residual sugar"] = st.slider(
            "Residual Sugar (g/dm³)",
            min_value=0.5, max_value=16.0,
            value=float(preset_data["residual sugar"]) if preset_data else 2.5,
            step=0.1,
            help="Natural sugar remaining after fermentation ceases."
        )

        input_values["chlorides"] = st.slider(
            "Chlorides (sodium chloride g/dm³)",
            min_value=0.01, max_value=0.60,
            value=float(preset_data["chlorides"]) if preset_data else 0.087,
            step=0.005,
            help="Amount of salt present in the wine."
        )

        input_values["density"] = st.slider(
            "Density (g/cm³)",
            min_value=0.990, max_value=1.004,
            value=float(preset_data["density"]) if preset_data else 0.9967,
            step=0.0005,
            help="Density depends on percent alcohol and residual sugar."
        )

        input_values["alcohol"] = st.slider(
            "Alcohol (% by volume)",
            min_value=8.0, max_value=15.0,
            value=float(preset_data["alcohol"]) if preset_data else 10.4,
            step=0.1,
            help="Alcoholic strength of the wine — strongest single predictor."
        )

    # Convert to DataFrame in exact feature order
    input_df = pd.DataFrame([input_values])[feature_names]

    # Apply training outlier bounds (guarantees safe inference consistency)
    input_processed = input_df.copy()
    for col in feature_names:
        if col in outlier_bounds:
            low_b, up_b = outlier_bounds[col]
            input_processed[col] = input_processed[col].clip(lower=low_b, upper=up_b)

    # Perform Prediction
    prediction = model.predict(input_processed)[0]
    probabilities = model.predict_proba(input_processed)[0]
    class_labels = list(model.classes_)  # ['High', 'Low', 'Medium'] or similar order

    prob_dict = dict(zip(class_labels, probabilities))

    # Prediction Presentation
    st.markdown("---")
    st.markdown("### Prediction Results")

    res_col1, res_col2 = st.columns([1, 1.4])

    with res_col1:
        if prediction == "High":
            card_class = "pred-high"
            tier_label = "HIGH QUALITY (Tier 1)"
            tier_desc = "Exquisite craftsmanship: rich alcohol body, optimal acidity, and balanced sulphates."
            badge_icon = "🌟"
            color_hex = "#22c55e"
        elif prediction == "Medium":
            card_class = "pred-medium"
            tier_label = "MEDIUM QUALITY (Tier 2)"
            tier_desc = "Good everyday table wine: balanced profile with moderate alcohol and typical acidity."
            badge_icon = "🍷"
            color_hex = "#f59e0b"
        else:
            card_class = "pred-low"
            tier_label = "LOW QUALITY (Tier 3)"
            tier_desc = "Below-standard profile: elevated volatile acidity or lower alcohol concentration."
            badge_icon = "⚠️"
            color_hex = "#ef4444"

        st.markdown(f"""
        <div class="prediction-card {card_class}">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">{badge_icon}</div>
            <div style="font-size: 1.6rem; font-weight: 700; color: {color_hex};">{tier_label}</div>
            <p style="color: #e2e8f0; font-size: 0.95rem; margin-top: 0.5rem;">{tier_desc}</p>
            <div style="margin-top: 1rem; font-size: 0.9rem; color: #cbd5e1;">
                Primary Class Confidence: <strong>{prob_dict[prediction]*100:.1f}%</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with res_col2:
        st.markdown("#### Class Probability Distribution")
        
        # High Quality Bar
        p_high = prob_dict.get("High", 0.0)
        st.markdown(f"**High Quality (Quality ≥ 7)**: `{p_high*100:.1f}%`")
        st.progress(float(p_high))

        # Medium Quality Bar
        p_med = prob_dict.get("Medium", 0.0)
        st.markdown(f"**Medium Quality (Quality = 6)**: `{p_med*100:.1f}%`")
        st.progress(float(p_med))

        # Low Quality Bar
        p_low = prob_dict.get("Low", 0.0)
        st.markdown(f"**Low Quality (Quality ≤ 5)**: `{p_low*100:.1f}%`")
        st.progress(float(p_low))

        st.info("💡 **Inference Insight:** Random Forest calculates probabilities across 300 decision trees. Because class weights are balanced, minority high-quality wines are treated with equal penalty weight.")

# -----------------------------------------------------------------------------
# TAB 2: Feature Importance & Insights
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### Predictive Importance vs Linear Correlation")
    st.markdown("""
    Evaluating whether linear correlation aligns with multivariate feature importance in this dataset.
    """)

    fi_data = metadata.get("feature_importances", {})
    if fi_data:
        fi_df = pd.DataFrame(list(fi_data.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)

        col_fi1, col_fi2 = st.columns([1.5, 1])

        with col_fi1:
            st.markdown("#### Random Forest Gini Feature Importance")
            st.bar_chart(fi_df.set_index("Feature")["Importance"], horizontal=True, color="#d946ef")

        with col_fi2:
            st.markdown("#### Top Predictive Drivers")
            st.markdown("""
            1. **Alcohol (17.8%)**: Highest Gini importance and strongest positive linear correlation ($r = +0.48$).
            2. **Sulphates (12.4%)**: Critical stabilizer preventing oxidation; higher sulphates strongly differentiate quality.
            3. **Volatile Acidity (11.6%)**: Negative driver. High acetic acid ruins sensory perception.
            4. **Total Sulfur Dioxide (10.9%)**: Key indicator of shelf-life and chemical equilibrium.
            """)
            st.success("✅ **Finding:** Alcohol had the highest absolute correlation with wine quality (|r| = 0.48) and the highest Random Forest feature importance (17.85%) in this dataset, followed synergistically by sulphates and volatile acidity.")

# -----------------------------------------------------------------------------
# TAB 3: Benchmark & Methodology
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### Model Benchmark Comparison (Test Set & 5-Fold CV)")
    
    benchmark_data = [
        {"Model": "Random Forest (Selected)", "Accuracy": "60.29%", "Test Macro F1": "0.5899", "5-Fold CV Macro F1": "0.6060 ± 0.0432", "High Class F1": "0.5600"},
        {"Model": "Gradient Boosting", "Accuracy": "61.76%", "Test Macro F1": "0.5919", "5-Fold CV Macro F1": "0.5870 ± 0.0263", "High Class F1": "0.5333"},
        {"Model": "Logistic Regression", "Accuracy": "59.19%", "Test Macro F1": "0.5872", "5-Fold CV Macro F1": "0.5811 ± 0.0300", "High Class F1": "0.6237"},
        {"Model": "K-Nearest Neighbors", "Accuracy": "58.82%", "Test Macro F1": "0.5591", "5-Fold CV Macro F1": "0.5496 ± 0.0344", "High Class F1": "0.5000"},
        {"Model": "Decision Tree", "Accuracy": "52.94%", "Test Macro F1": "0.5142", "5-Fold CV Macro F1": "0.5452 ± 0.0226", "High Class F1": "0.4565"}
    ]
    st.dataframe(pd.DataFrame(benchmark_data), use_container_width=True, hide_index=True)

    st.markdown("### Model Selection Rationale")
    st.markdown("""
    Random Forest was selected as the final model because it achieved the highest 5-fold cross-validated Macro F1 (**0.6060 ± 0.0432**) and strong minority-class performance (**High F1: 0.5600**). Gradient Boosting achieved slightly higher Macro F1 on the single test split (0.5919 vs 0.5899).
    """)

    st.markdown("### Why Macro F1 Over Accuracy?")
    st.markdown("""
    The dataset exhibits notable class imbalance:
    - **Low Quality (≤ 5):** 47.1% of samples
    - **Medium Quality (= 6):** 39.4% of samples
    - **High Quality (≥ 7):** 13.5% of samples

    A naive classifier predicting only 'Low' and 'Medium' could achieve high overall accuracy while having **zero utility** for identifying rare high-quality wines.
    **Macro F1** gives equal weight to all three classes, ensuring that identifying high-quality wine is rewarded appropriately.
    """)
