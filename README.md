# 📊 Modern Telecom Customer Churn Analysis & Prediction

This project provides an end-to-end data science solution for identifying and predicting customer churn in the telecommunications industry. It features a comprehensive **exploratory data analysis (EDA)** and a **Streamlit-based web application** that allows stakeholders to predict churn risk for individual customers using trained Machine Learning models.

## 🚀 Live Dashboard Features

The interactive dashboard is organized into five key phases:

* **Dataset Overview**: High-level metrics including total customers, overall churn rate, and statistical summaries of the dataset.
* **Univariate Analysis**: Detailed distribution analysis of individual features like tenure, contract types, and monthly charges.
* **Bivariate Analysis**: Deep dives into relationships, such as how contract types or payment methods directly correlate with churn.
* **Multivariate Analysis**: A correlation matrix visualizing the strength of relationships between numerical variables and churn.
* **Churn Prediction Model**: A real-time predictor where users can input customer demographics and service details to get an instant risk assessment.

## 🧠 Machine Learning Approach

The application utilizes two primary models to ensure robust predictions:

* **Gradient Boosting (Champion Model)**: Optimized for the best F1-score and recall to ensure high-risk customers are accurately identified.
* **Logistic Regression**: Provides a balanced, interpretable alternative for risk assessment.
* **Preprocessing**: Includes automated scaling for numerical features (tenure, charges) and one-hot encoding for categorical data to ensure model accuracy.

## 🛠️ Tech Stack

* **Language**: Python
* **Libraries**: Pandas, NumPy, Scikit-Learn, Imbalanced-Learn
* **Visualization**: Plotly, Matplotlib, Seaborn
* **Deployment**: Streamlit

## 📁 Project Structure

```text
├── Dataset/          # Raw and cleaned customer data
├── Models/           # Saved .pkl files (Models, Scaler, Column mappings)
├── NoteBooks/        # Jupyter notebooks for cleaning, EDA, and model building
├── app/              # Streamlit web application source code
└── requirements.txt  # Project dependencies

```

## ⚙️ Installation & Usage

1. **Clone the repository**:
```bash
git clone https://github.com/[Rohit-Girhe]/Customer-Churn.git

```


2. **Install dependencies**:
```bash
pip install -r requirements.txt

```


3. **Run the Dashboard**:
```bash
streamlit run app/webapp.py

```



## 💡 Key Business Insights Derived

* **The Contract Trap**: Month-to-month users exhibit a significantly higher churn rate (~42.7%) compared to long-term contract holders.
* **Billing Friction**: Customers using electronic checks as a payment method are at a higher risk, churning at rates over 45%.
* **Retention Strategy**: Analysis suggests implementing a "First-Year Loyalty Program" as the median tenure for churners is approximately 10 months.