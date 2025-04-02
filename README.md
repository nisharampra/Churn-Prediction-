# Churn-Prediction-

# 📊 Customer Churn Prediction

This project aims to predict customer churn using machine learning techniques on the [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn). The goal is to help businesses identify customers who are likely to leave, enabling proactive retention strategies.

---

## 🎯 Project Objective

Predict whether a customer will **churn** (i.e., discontinue service) based on features such as demographics, service usage, and billing information.

---

## 🧰 Tools & Technologies

- **Python**  
- **Jupyter Notebook**
- **Libraries:** pandas, numpy, seaborn, matplotlib, scikit-learn

---

## 📁 Dataset

- Source: Kaggle ([Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn))
- 7,043 customer records
- Features include:
  - Customer demographics
  - Services subscribed (internet, phone, etc.)
  - Billing and payment information
  - Churn status (target)

---

## ⚙️ Workflow

### 1. Data Preprocessing
- Handled missing values
- Converted `TotalCharges` to float
- Encoded categorical features using one-hot encoding
- Standardized numerical features

### 2. Modeling
- Trained two models:
  - **Logistic Regression** (baseline)
  - **Random Forest** (ensemble model)

### 3. Evaluation Metrics
- Accuracy
- ROC AUC Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1)

---

## 📈 Results

| Model               | Accuracy | ROC AUC | F1 Score (Churn) |
|--------------------|----------|---------|------------------|
| Logistic Regression| ~78%     | 0.70    | 0.56             |
| Random Forest       | ~78%     | 0.69    | 0.54             |

- Logistic Regression performed slightly better in detecting churn.
- Class imbalance affected recall for the churn class.

---

## 🔮 Sample Prediction

The model can predict churn for a new customer based on their data.

```python
sample = X_test.iloc[0:1]
rf.predict(sample)



Future Improvements
Handle class imbalance (e.g., SMOTE, class weights)

Use XGBoost or LightGBM for higher performance

Hyperparameter tuning with GridSearchCV

Deploy with Streamlit or Flask for real-time predictions

