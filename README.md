# 🍷 Red Wine Quality Classification — Machine Learning Case Study

An end-to-end Machine Learning case study predicting artisanal red wine quality tiers from physico-chemical properties. Features comprehensive exploratory data analysis, leak-free stratified cross-validation, 5-model comparative benchmarking, feature importance interpretation, and an interactive Streamlit web application.

---

## 📌 Project Overview

- **Dataset:** Portuguese "Vinho Verde" Red Wine Dataset (`data/winequality-red.csv`)
- **Instances:** 1,599 raw observations (1,359 unique rows after deduplication)
- **Features:** 11 continuous physico-chemical measurements
- **Objective:** Multi-class classification into three quality categories:
  - **Low Quality (Tier 3):** Wine sensory score $\le 5$ (47.1% of samples)
  - **Medium Quality (Tier 2):** Wine sensory score $= 6$ (39.4% of samples)
  - **High Quality (Tier 1):** Wine sensory score $\ge 7$ (13.5% of samples)

---

## 📊 5-Model Benchmark & Selection

All models were evaluated on an 80/20 stratified holdout test set ($N=272$) and via 5-Fold Stratified Cross-Validation on the training set ($N=1,087$):

| Model | Test Accuracy | Test Macro F1 | 5-Fold CV Macro F1 | High Class F1 | Selected |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Selected)** | **60.29%** | **0.5899** | **0.6060 ± 0.0432** | **0.5600** | **✅ FINAL** |
| **Gradient Boosting** | 61.76% | 0.5919 | 0.5870 ± 0.0263 | 0.5333 | — |
| **Logistic Regression** | 59.19% | 0.5872 | 0.5811 ± 0.0300 | 0.6237 | — |
| **KNN (k=7)** | 58.82% | 0.5591 | 0.5496 ± 0.0344 | 0.5000 | — |
| **Decision Tree** | 52.94% | 0.5142 | 0.5452 ± 0.0226 | 0.4565 | — |

### Key Selection Criteria:
1. **Macro F1 Over Accuracy:** Because only 13.5% of samples belong to the "High" quality tier, overall accuracy is inflated by majority classes. Macro F1 weights all three tiers equally.
2. **Model Selection Rationale:** Random Forest was selected as the final model because it achieved the highest 5-fold cross-validated Macro F1 (**0.6060 ± 0.0432**) and strong minority-class performance (**High F1: 0.5600**). Gradient Boosting achieved slightly higher Macro F1 on the single test split (0.5919 vs 0.5899).
3. **Minority Class Performance:** Random Forest achieved balanced per-class scores (`Low: 0.6800`, `Medium: 0.5297`, `High: 0.5600`), successfully identifying high-tier wines without ignoring medium-tier wines.

---

## 🔬 Key Scientific Findings

- **Feature Importance & Correlation Insights:**
  - Alcohol had the highest absolute correlation with wine quality ($|r| = 0.4803$) and the highest Random Forest feature importance ($17.85\%$) in this dataset.
  - However, correlation and feature importance reflect statistical association rather than causation. Model analysis reveals non-linear synergies: high alcohol must be balanced with low **volatile acidity** (to avoid vinegary sensory notes) and sufficient **sulphates** (antioxidant/preservative action).

---

## 🛠️ Project Structure

```text
winequalityml/
├── data/
│   └── winequality-red.csv         # Raw Portuguese Vinho Verde dataset
├── models/
│   ├── model.pkl                   # Trained Random Forest Classifier
│   ├── scaler.pkl                  # Fitted StandardScaler
│   ├── feature_names.pkl           # List of 11 feature names
│   ├── outlier_bounds.pkl          # IQR clipping bounds computed on X_train
│   └── model_metadata.json         # Complete performance & feature statistics
├── notebooks/
│   └── wine_quality_analysis.ipynb # 61-cell notebook (Phases 1-10 complete)
├── src/
│   └── train_and_save.py           # Reproducible model training & serialization script
├── app.py                          # Modern Streamlit UI for real-time inference
├── requirements.txt                # Pinned production dependencies
└── README.md                       # Case study documentation
```

---

## 🚀 How to Run the Project

### 1. Environment Setup
```bash
# Clone and enter directory
cd winequalityml

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to test custom wine profiles, inspect probability bars, and view model benchmarks.

### 3. Re-train & Serialize Models
```bash
python src/train_and_save.py
```

### 4. Open the Analysis Notebook
```bash
jupyter notebook notebooks/wine_quality_analysis.ipynb
```
*(Select the **Python (Wine Quality)** kernel)*
