# 🌱 CottonSeed AI — Cotton Seed Quality Prediction Intelligence Model

<div align="center">
<img src="Cotton Plantation.jpg" alt="Cotton Field" width="70%" style="border-radius: 12px; max-height: 320px; object-fit: cover;" />

  A next-generation precision intelligence platform that autonomously predicts cotton seed germination quality, classifies vigor class, and delivers agronomic insights — from raw field data, in seconds.
</div>

## 📸 Application Preview

![App Screenshot](./assets/app_screenshot.png)

### 🎥 Demo Video
> *Short walkthrough of the CottonSeed AI application — prediction, batch processing, and analytics dashboard*

[![Watch Demo](https://img.shields.io/badge/Watch-Demo-blue)](./assets/demo_video.mp4)

🔗 **Live App:** https://cotton-seed-quality-prediction.streamlit.app/

---

## 📌 Project Overview

### 🧩 Business Challenge
Cotton seed quality assessment has historically been:
- Manual  
- Reactive (post-harvest testing)  
- Fragmented across systems  

This leads to:
- Poor seed lot classification  
- Yield losses and replanting costs  
- Supply chain inefficiencies  
- Mispriced inventory and financial losses  

---

### 💡 Rationale for the Project
The cotton industry spans **agriculture, textiles, and food systems**, powering a **$600B+ global value chain**.  

Cotton seed is a critical asset:
- 🌾 Cottonseed oil (food industry)  
- 🐄 Livestock feed (meal & hulls)  
- 🧵 Textile and industrial by-products  

Yet, **seed quality determines the success of the entire value chain**.

---

### 🎯 Project Objectives
This project aims to:

- Predict **seed germination performance** before planting  
- Classify seed lots into **vigor categories**  
- Provide **data-driven insights for decision-making**  
- Enable **scalable, automated seed quality evaluation**  

---

## 🧠 Key Concepts in Cotton Seed Quality

### 🌡️ Warm Germination (WG)
- Measures germination under **ideal conditions**
- Indicates **maximum viability potential**

### ❄️ Cold Test (CT)
- Measures germination under **cold stress conditions**
- Indicates **seed vigor and field performance**

---

### 📊 Derived Vigor Indices
- **CWVI (Cool-Warm Vigor Index)** = WG + CT  
- **Vigor Gap** = WG − CT  
- **CT/WG Ratio** = Cold tolerance efficiency  

These metrics determine:
- 🌟 Excellent  
- 👍 Good  
- ⚠️ Poor seed lots  

---

## 📈 Model Performance & Evaluation

### Final Production Models

| Target | Algorithm | Tuning | Test R² | Val R² | Test RMSE | Training Data |
|---|---|---|---|---|---|---|
| Warm Germination (WG) | XGBoost | Optuna TPE (120 trials) | **0.9608** | 0.9566 | 1.621 pts | Original splits |
| Cold Test (CT) | GradientBoosting | Baseline (ceiling already met) | **0.9990** | 0.9989 | 0.782 pts | SMOTE-augmented |

### Understanding the Performance Gap Between WG and CT

The CT model's near-perfect R² (0.9990) compared to WG (0.9608) reflects a fundamental agronomic difference between the two targets:

**Cold Test (CT)** is primarily driven by a small number of high-signal features — most importantly `ct_state_prior` (the regional cold-tolerance baseline derived from historical lot data) and physical seed quality metrics (boll opening percentage, cold stress index, seed coat integrity). Once the geographic-agronomic baseline is correctly encoded, CT variance reduces to a narrow, predictable band. This makes it highly modelable.

**Warm Germination (WG)** is influenced by a wider, noisier set of interacting factors — storage duration, moisture at test time, mechanical damage, harvest timing, and laboratory variation across test facilities. This inherent variability means a lower (but still excellent) R² represents genuine irreducible noise, not model inadequacy.

### Class Imbalance & SMOTE

The training dataset contained a severe class imbalance — only **7.6% of lots qualified as Excellent Vigor**. Without correction, the CT model would default to predicting the majority (Poor/Good) class and systematically under-predict high-quality lots.

**SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the CT training data to increase the Excellent Vigor representation to **20% of the training set** (+244 synthetic lots). SMOTE was applied *exclusively* to training data — validation and test sets contained only real lots to ensure honest evaluation. This resolved the prediction suppression issue entirely: the deployed model correctly identifies Excellent Vigor lots at ~8% of test predictions, consistent with real-world prevalence.

### 📌 Interpretation
- WG model explains **96% of variability**
- CT model explains **99.9% of variability**
- Both models show **high predictive reliability on unseen data**

---

## 🧪 Model Validation & Generalization

### Three-Way Holdout Design

Data was split into three non-overlapping sets:

| Set | Size | Purpose |
|---|---|---|
| Training | ~65% | Model fitting |
| Validation | ~15% | Hyperparameter tuning, early stopping |
| Test | ~20% (411 lots) | Final honest evaluation — never seen during training or tuning |

SMOTE was applied *only* to training data. Validation and test sets contain exclusively real, original lots — ensuring that reported test metrics reflect genuine generalisation to unseen production data.

### ✅ Generalisation Proof — Val/Test R² Gap

The key metric that proves a model will perform consistently on new data is the gap between its validation R² (seen during tuning) and its test R² (never seen at all). A large gap signals overfitting; a small gap signals genuine learning.

| Model | Val R² | Test R² | Gap | Assessment |
|---|---|---|---|---|
| XGBoost WG | 0.9566 | 0.9608 | +0.0042 | ✅ Excellent generalisation |
| GradientBoosting CT | 0.9989 | 0.9990 | +0.0001 | ✅ At performance ceiling |

- Validation R² ≈ Test R²

➡️ Both gaps are well under 0.005 — the threshold used to confirm that model performance on completely fresh, never-seen data will match evaluation-time performance statistically. This means the models are expected to score new commercial seed lots from any U.S. growing region with the same precision demonstrated on the test set.

---

## 🔬 Feature Engineering & Data Integrity

### Dataset Overview

- **1,523 seed lots** from commercial U.S. cotton production
- **80+ raw features** spanning 6 agronomic domains
- **50 features selected** for final model training
- Coverage: multiple states (TX, CA, AZ, AR, MS, GA, and others), varieties, seasons (2019–2023), and irrigation systems

### Six Feature Domains

| Domain | Features Include |
|---|---|
| 🗓️ Field & Harvest | Planting/harvest DOY, season length, boll open-to-harvest window, defoliation timing |
| 🌾 Variety & Setup | Origin region, irrigation type, maturity rating, trait stack, macro zone |
| 🌡️ Weather & Thermal | Cumulative DD60, daily heat units, pre-plant soil moisture, defoliation heat accumulation |
| 🌱 Seed Physical Quality | Moisture %, seeds/lb, Xdamage, Xmaturity, cut test scores, FFA, mechanical damage |
| 🧪 Lot Testing | WG/CT initial scores, tetrazolium scores, ABA level, NAWF average |
| 🌿 Crop Phenology | Boll opening %, NACB average, yield, module acceptance rate, thermal efficiency |

---

### 🚫 **Data Leakage Prevention**

- Removed features containing **future/post-outcome information**
- Eliminated **target-derived variables**

**Data Leakage** — where information from the future or from the target variable contaminates the training features — was rigorously controlled at three levels:

**1. Temporal leakage**: Any feature that encodes post-harvest outcomes (e.g., final lot disposition, buyer acceptance flags, retest scores used to generate a final quality decision) was identified and excluded. The model is trained only on data that would realistically be available *at the time of prediction*.

**2. Target-derived leakage**: Features directly computed from WG or CT (such as the final vigor class label used as an input) were never included in the feature set. All composite indices are computed *from* the model's output — never fed into it.

**3. Geographic proxy leakage**: During diagnostic investigation, `pp_day10_avg_soilmoisture` was identified as a spurious geographic proxy — it encoded *which state the lot came from* (CA lots had low values and high CT; TX lots had high values and low CT) rather than a causal agronomic relationship. This feature's SHAP value was inverted relative to agronomic expectation, confirming it was acting as a geographic surrogate. It was replaced with the explicit, causally valid `ct_state_prior` feature.

---

### 📉 Feature Selection & Optimization

> ***Redundancy & Multicollinearity Removal***

- **Pearson Correlation** — Pairwise correlation matrices across all 80+ features identified collinear pairs with |r| > 0.90. For each collinear pair, the agronomically less meaningful feature was removed. This step eliminated 12 redundant features.

- **Variance Inflation Factor (VIF)** — Remaining features were assessed for multicollinearity using VIF analysis. Features with VIF > 10 were iteratively removed until the feature matrix stabilised, ensuring that no feature's variance was substantially explained by a linear combination of others.

- **Mutual Information (MI)** — MI scores ranked features by their non-linear relationship to WG and CT targets separately, capturing threshold effects and interaction patterns invisible to Pearson correlation. CT-specific features were force-protected from elimination via a dedicated `CT_FORCE_INCLUDE` list, preventing WG-correlated features from crowding them out during joint selection.


---

### 📊 Final Feature Set
- Reduced from **90+ → <50 features**
- Optimized for:
  - Interpretability  
  - Stability  
  - Performance  

---

## ⚙️ Application Capabilities

CottonSeed AI integrates six fully functional modules into a single production-grade web application:

| Module | Description |
|---|---|
| 🔬 **Single Lot Prediction** | Enter raw field and lab data for any seed lot → receive WG, CT, CWVI, Vigor Gap, CT/WG ratio, vigor class, and quality flags instantly |
| 📦 **Batch Prediction** | Upload a CSV of any number of lots → download a fully scored results file with all metrics |
| 📊 **Analytics Dashboard** | Interactive charts: regional performance maps, seasonal trends, variety comparisons, quality distributions with full hover/filter capability |
| 🔍 **Data Explorer** | SQL-style querying and multi-variable filtering across all 1,500+ lot records with CSV export |
| 📖 **About & Benchmarks** | Model specs, quality threshold guide, CWVI interpretation tables, agronomic context for every metric |
| 🧠 **SHAP Explainability** | Every prediction backed by ranked feature attribution — telling users not just *what* was predicted but *why* |

---

## ⚙️ Tech Stack

- **Programming:** Python  
- **Data Processing:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn, Plotly  
- **Machine Learning:** Scikit-learn, XGBoost, LightGBM, Ensemble - GradientBoosting
- **Hyperparameter Tuning** Optuna (TPE sampler)
- **Class Balancing** imbalanced-learn (SMOTE)
- **Explainability:** SHAP, LIME  
- **Model Tuning:** Optuna  
- **Deployment:** Streamlit  
- **Serialization:** Joblib
- **Version Control** GitHub

---

## 🗂️ Project Structure

```
AI-Powered-Cotton-Seed-Quality-Prediction-Model/
│
├── app.py                          # Main Streamlit application (6 integrated modules)
├── pipeline.py                     # CottonSeedQualityPipeline inference class
├── requirements.txt                # Pinned package versions for reproducible deployment
├── model_registry.json             # Deployed model names and performance metrics
├── selected_features.json          # 50 selected feature names
├── wg_engineered_features.json     # WG-specific engineered feature list
├── 03_clean_model_ready.parquet    # Clean dataset (powers Analytics Dashboard + Data Explorer)
│
├── .streamlit/
│   └── config.toml                 # App theme (dark navy) and server configuration
│
├── deployment_bundle/
│   └── cotton_seed_quality_bundle.joblib   # Full deployment bundle:
│                                           #   preprocessor + WG model + CT model +
│                                           #   SHAP explainers + LIME params +
│                                           #   feature names + label encoder
│
├── README.md                       # This file
└── LICENSE                         # MIT License
```

----


## 🌍 Industry Impact & Value

CottonSeed AI delivers value across the entire ecosystem:

### 🌾 Farmers
- Improved planting decisions  
- Reduced replanting risk  

### 🏭 Seed Companies
- Automated quality classification  
- Inventory optimization  

### 💼 Investors
- Risk-informed decisions  

### 🧪 Researchers
- Data-driven agronomic insights  

### 🧵 Supply Chain Players
- Stable and predictable seed quality  

---

## 📈 Why This Matters

A single misclassified seed lot can result in:
- Yield loss  
- Supply chain disruption  
- $50K–$200K+ financial impact  

CottonSeed AI transforms:
> ❌ Reactive testing  
> into  
> ✅ Predictive intelligence  

---

## 🔮 Future Enhancements

- Real-time weather API integration  
- Satellite data (NDVI, soil maps)  
- Deep learning models (time-series)  
- API deployment for enterprise systems  

---

## 🤝 Contribution & Collaboration

Contributions, feedback, and collaborations are welcome.

---

## 📬 Contact

**Ndifreke Ekanem**  🔗

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/NdifrekeAnalytics/) [![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=maildotru&logoColor=white)](mailto:sync872000@gmail.com) [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ndifreke-ekanem-b479a027/)

---

## ⭐ Acknowledgment

Special thanks to **10Alytics** and **AMDARI** for foundational training and support.

---

🌱 *Turning agricultural data into intelligent, scalable decisions.*
