# 📊 Modern Telecom Churn Prediction & Analytics Dashboard

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Deployment](https://img.shields.io/badge/Deployment-Streamlit-red)
![Data Visualization](https://img.shields.io/badge/Data%20Visualization-Plotly-green)

## 📌 Project Overview
Customer churn is a critical metric for telecommunication companies. Retaining an existing customer is significantly cheaper than acquiring a new one. This project is an end-to-end Data Science and Machine Learning solution designed to analyze telecom customer data, uncover actionable business insights, and predict the likelihood of customer churn. 

The final deliverable is a fully interactive, highly optimized **Streamlit Web Dashboard** that allows stakeholders to explore the data visually and predict individual customer churn risk in real-time using a streamlined, 8-question interface.

---

## 🏗️ Technical Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Plotly Express, Seaborn, Matplotlib
* **Machine Learning:** Scikit-Learn, Imbalanced-Learn (SMOTE)
* **Web Deployment:** Streamlit

---

## 🔬 The Data Science Pipeline

### 1. Data Cleaning & Feature Engineering
* Handled missing values and standardized categorical text.
* **Multicollinearity Fix:** Strictly dropped the `TotalCharges` feature, as it was highly correlated with `tenure` and `MonthlyCharges`, ensuring a mathematically sound model.

### 2. Advanced Feature Selection (Mutual Information)
To ensure the final deployed application was fast, user-friendly, and resistant to overfitting, a rigorous statistical test was performed.
* Applied **Mutual Information (MI)** to calculate the exact mathematical dependency between all 22 input features and the `Churn` target variable.
* **The Result:** Sliced the dataset down by 60%, retaining only the **Top 8 Master Features** (`Contract`, `tenure`, `PaymentMethod`, `MonthlyCharges`, `InternetService`, `PaperlessBilling`, `Partner`, `Dependents`).

### 3. Targeted Standardization & Class Balancing
* Applied `StandardScaler` **strictly to continuous variables** (`tenure` and `MonthlyCharges`) to prevent distorting binary encoded categorical features. 
* The original dataset was highly imbalanced (~73% Retained / 27% Churned). Applied **SMOTE (Synthetic Minority Over-sampling Technique)** strictly to the training data to create a perfectly balanced 50/50 dataset.

### 4. Blind Algorithm Tournament
To ensure a rigorously fair experiment, multiple algorithms were trained on the optimized 8-feature dataset and evaluated on an unseen test set. 

**🏆 The Champion Model:**
* **Gradient Boosting (GBDT)** successfully defended its title as the most balanced model.
* **F1-Score:** ~0.62 
* **Recall:** ~0.78 (Successfully catches nearly 80% of all churning customers)
* *Note: Despite dropping 14 columns of data, the model retained over 95% of its predictive power compared to the heavy 22-feature baseline, proving it is highly optimized for production.*

---

## 💻 Streamlit Web Application
The predictive models and EDA were deployed into a sleek, multipage Streamlit dashboard. 

### Dashboard Features:
1. **Dataset Overview:** High-level metrics and statistical summaries.
2. **Univariate Analysis:** Interactive distributions of individual features with strategic business insights.
3. **Bivariate Analysis:** Cross-feature comparisons (e.g., Contract Type vs. Churn Rate) to identify high-risk segments.
4. **Multivariate Analysis:** Correlation heatmaps to map feature relationships.
5. **🔮 Streamlined Churn Predictor:** A real-time inference engine. By utilizing our 8-feature lightweight model, sales reps and customer service agents only need to input the most critical data points to get an instant, highly accurate Churn Probability Score.

---

## 🚀 How to Run Locally

**1. Clone the repository**
```bash
git clone [https://github.com/Rohit-Girhe/telecom-churn-dashboard.git](https://github.com/Rohit-Girhe/telecom-churn-dashboard.git)
cd telecom-churn-dashboard