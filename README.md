# Creditwise Loan System (CWLS)

An end-to-end machine learning pipeline designed to automate and optimize the credit underwriting process. This system evaluates loan applicant data to predict approval viability, explicitly optimizing for financial risk mitigation and capital protection.

---

## 📌 Project Overview
Manual credit risk assessment is time-consuming and prone to human bias. The **Creditwise Loan System** leverages predictive analytics to instantly assess an applicant's profile. 

By analyzing key financial and demographic indicators, the system predicts whether a loan application should be approved or rejected. The core objective of this project is to prioritize **risk aversion**, minimizing non-performing assets (NPAs) and bad debt allocations.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python 3.x
* **Environment:** JupyterLab / Jupyter Notebook
* **Core Libraries:**
  * `pandas` & `numpy` — Data manipulation and feature engineering
  * `scikit-learn` — Model training, scaling, and evaluation metrics

---

## 📊 Pipeline Workflow

### 1. Exploratory Data Analysis & Preprocessing
* Handled feature constraints and dropped unnecessary tracking columns.
* **Feature Scaling:** Implemented `StandardScaler` to normalize continuous variables, ensuring distance-based and optimization-based algorithms (like KNN and Logistic Regression) converge efficiently without scaling bias.

### 2. Feature Engineering
* Captured complex, non-linear financial relationships by generating polynomial squared transformations for key ratios and metrics:
  * `DTI_Ratio_sq` ($DTI\_Ratio^2$)
  * `Credit_Score_sq` ($Credit\_Score^2$)

### 3. Model Training & Evaluation
The project explicitly benchmarks different inductive biases against each other to find the most robust classifier:
* **Logistic Regression:** A linear model optimized for binary classification.
* **Gaussian Naive Bayes:** A baseline probabilistic model assuming feature independence.
* **K-Nearest Neighbors (KNN):** A distance-based neighborhood approach optimized at $k=17$.

---

## 📈 Model Performance Benchmark

The models were evaluated using standard classification metrics: **Precision, Accuracy, Recall, and F1-Score** alongside their respective Confusion Matrices.

| Model | Precision | Accuracy | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Gaussian Naive Bayes (Production Choice)** | **81.1%** | 86.0% | 70.5% | 75.4% |
| **Logistic Regression** | 78.5% | **88.0%** | **83.6%** | **81.0%** |
| **KNN ($k=17$)** | 76.2% | 80.5% | 52.5% | 62.1% |

---

## 🎯 Strategic Business Insights & Model Selection

### ⚖️ The Precision vs. Recall Trade-off in Credit Underwriting
In fintech and banking, a framework cannot be evaluated purely on vanilla accuracy. The economic cost of model errors is asymmetrical:
* **False Positive (Low Precision):** Approving a loan for a high-risk applicant who will ultimately default. This results in direct **capital destruction** (NPA).
* **False Negative (Low Recall):** Rejecting a qualified applicant. This results in a **lost opportunity cost** (missed interest revenue), but preserves core capital.

### 🏆 Why Gaussian Naive Bayes is the Champion Model
Despite Logistic Regression having a higher overall accuracy (88.0%), **Gaussian Naive Bayes is selected for production** due to its superior **Precision (81.1%)**.

* **Capital Protection:** Naive Bayes minimizes False Positives better than any other benchmarked model. It ensures that when the system flags a loan as "Approved", there is a higher mathematical certainty of repayment.
* **Risk-Averse Alignment:** For conservative financial institutions prioritizing portfolio health and low default rates over aggressive customer acquisition, Naive Bayes offers the safest deployment profile.

---

## 🚀 How to Run the Project

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/AICatalyst/Creditwise-Loan-System.git](https://github.com/AICatalyst/Creditwise-Loan-System.git)
   cd Creditwise-Loan-System