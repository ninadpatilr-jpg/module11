
# Used Car Price Prediction  
Analyzing 400k+ used car listings to understand **which factors impact price the most**, perform **data cleaning**, **feature engineering**, **EDA**, and build **ML models** (Ridge, RandomForest, XGBoost) to predict car prices.

---

# Project Overview

This project explores a large Kaggle dataset containing nearly **426,000 used car listings**.  
The objectives are:

1. **Identify which features influence car price the most**
2. **Clean and prepare the dataset** for machine learning  
3. **Engineer powerful new features** (car age, miles per year, condition score, etc.)  
4. **Perform EDA** to visualize key relationships (price vs year, odometer, manufacturers…)  
5. **Train predictive models** using:
   - Linear Regressor
   - Ridge Regression
   - RandomForestRegressor

The final goal is to build a reliable and interpretable pricing model.

---

# Data Preparation

### ✔ Steps Performed
- Removed irrelevant columns (`id`, `VIN`, `region`)
- Filtered out incorrect price values (`price < 500`)
- Log-transformed price → `price_log`
- Created engineered features:
  - `car_age = 2025 - year`
  - `miles_per_year = odometer / car_age`
  - `cylinders_num` extracted from `"6 cylinders"`
  - `condition_num` (ordinal mapping)
- Frequency-encoded high-cardinality columns (`model`, `region`)
- Imputed missing values (median for numeric, most_frequent for categorical)
- One-hot encoded categorical variables (sparse to preserve RAM)
- Standardized numerical features

A memory-safe sample (`50,000 rows`) is used for modeling to avoid crashing local machines.

---

### **Correlation Heatmap**
Important correlations:
- `price` ↓ with `year`
- `price` ↓ with `odometer`
- `price` ↓ with `cylinders_num`
- `price` ↑ with `condition_num`

---

# Modeling

Three models were used:

---
## **1. Linear Regressor**
## **2. Ridge Regression**
## **3. RandomForest Regressor**

---

# Most Important Features Across All Models

| Rank | Feature | Description |
|------|---------|-------------|
| **1** | `car_age` | Newer cars = higher price |
| **2** | `odometer` | Mileage strongly impacts depreciation |
| **3** | `cylinders_num` | Higher engine power = higher cost |
| **4** | `fuel` | Diesel cars priced higher |
| **5** | `manufacturer` | Luxury vs budget brands |
| **6** | `drive` | AWD/4WD more expensive |

---

# How to Run This Project

### Run the notebook
```bash
jupyter notebook used_car_price_analysis.ipynb
```

