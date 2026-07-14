# Creditwise Loan System (CWLS)

An end-to-end machine learning pipeline designed to automate and optimize the credit underwriting process. This system evaluates loan applicant data to predict approval viability, explicitly balancing risk management with portfolio growth.

---

## 📌 Project Overview

Manual credit risk assessment is time-consuming and prone to human bias. The **Creditwise Loan System (CWLS)** leverages predictive analytics to instantly assess an applicant's financial profile. By evaluating critical indicators like credit history, debt obligations, and demographics, the system predicts whether a loan application should be approved or rejected.

The primary goal of this framework is to maximize precision and reliability, thereby minimizing non-performing assets (NPAs) and mitigating financial risk for credit providers.

---

## 🛠️ Tech Stack & Libraries

* **Language:** Python 3.x
* **Environment:** JupyterLab / Jupyter Notebook
* **Core Libraries:**
* `pandas` & `numpy` — Data preprocessing, cleansing, and manipulation
* `scikit-learn` — Feature scaling, model training, and performance metrics


---

## 📊 Pipeline Workflow

### 1. Exploratory Data Analysis & Preprocessing

* Checked and managed missing entries and handled feature constraints.
* **Feature Scaling:** Implemented `StandardScaler` to normalize continuous variables (such as applicant income, asset valuations, and loan amounts) ensuring distance-based and optimization-based models converge efficiently without feature scaling bias.

### 2. Feature Engineering

* Captured complex, non-linear financial interactions by generating polynomial squared transformations for critical baseline variables:
* Debt-to-Income impact squared (`DTI_Ratio_sq`)
* Applicant credit worthiness squared (`Credit_Score_sq`)

### 3. Model Training & Evaluation

The project benchmarks different inductive biases against one another to discover the most robust framework for binary credit classification:

* **Logistic Regression:** A linear model optimized for binary classification.
* **Gaussian Naive Bayes:** A baseline probabilistic model assuming conditional independence.
* **K-Nearest Neighbors (KNN):** A distance-based neighborhood classifier optimized at $k = 17$.

---

## 📈 Model Performance Benchmark

The models were evaluated using standard classification metrics: **Precision, Accuracy, Recall, and F1-Score** derived alongside their respective validation datasets.

| Model | Precision | Accuracy | Recall | F1-Score |
| --- | --- | --- | --- | --- |
| **Logistic Regression (Production Choice)** | **79.0%** | **87.5%** | **80.3%** | **79.7%** |
| **Gaussian Naive Bayes** | 78.3% | 86.5% | 77.0% | 77.7% |
| **KNN ($k = 17$)** | 70.3% | 77.0% | 42.6% | 53.1% |

---

## 🎯 Strategic Business Insights & Model Selection

### ⚖️ The Precision vs. Recall Trade-off in Credit Underwriting

In financial lending, the economic cost of prediction errors is highly asymmetrical:

* **False Positive (Low Precision):** Approving a loan for a high-risk applicant who will ultimately default. This results in direct **capital destruction** (bad debt).
* **False Negative (Low Recall):** Rejecting a qualified applicant. This results in a **lost opportunity cost** (missed interest revenue), but preserves core capital.

### 🏆 Why Logistic Regression is the Champion Model

Following our feature engineering transformations, **Logistic Regression** emerged as the optimal deployment choice:

* **Dominant Performance:** It outperforms alternative benchmarks across the board, securing the highest overall **Accuracy (87.5%)** and **Recall (80.3%)**.
* **Capital Protection:** By producing the highest baseline **Precision (79.0%)**, it minimizes False Positives better than Naive Bayes or KNN, ensuring that approved loans carry the highest mathematical probability of repayment.

---

## 🚀 How to Run the Project

1. **Clone this repository:**
```bash
git clone https://github.com/AICatalyst890/Creditwise-Loan-System.git

```

2. **Navigate to the project folder:**
```bash
cd Creditwise-Loan-System

```

3. **Install dependencies:**
```bash
pip install -r requirements.txt

```

4. **Open and run the Jupyter Notebook:**
```bash
jupyter lab cwls.ipynb

```

---

## 📂 Repository Structure

* `cwls.ipynb` — Full Jupyter notebook containing EDA, preprocessing, feature engineering, and model training.
* `loan_approval_data.csv` — Raw dataset containing applicant financial and demographic profiles.
* `requirements.txt` — File detailing project environmental dependencies.
* `README.md` —  Document detailing the project overview, workflow, benchmarks, and running instructions.