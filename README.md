# 🏦 Loan Approval Prediction using Machine Learning

## 📌 Project Overview
This project focuses on building a **Machine Learning model** that predicts whether a loan application will be approved based on various applicant details.  
Loan approval is a critical process for banks and financial institutions, and predictive analytics can help automate and speed up this process.

---

## ✅ Problem Statement
Banks receive thousands of loan applications daily. Manual verification takes time and resources, and wrong decisions can lead to losses.  
**Goal:** Use historical loan application data to predict whether a new loan should be **Approved** or **Rejected**.

---

## 🛠 Steps Taken:
1️⃣ **Data Collection & Understanding**
- Loaded the dataset containing details like gender, income, credit history, loan amount, etc.

2️⃣ **Data Cleaning & Preprocessing**
- Handled missing values.
- Encoded categorical variables.
- Scaled numerical features for better model performance.

3️⃣ **Exploratory Data Analysis (EDA)**
- Visualized distributions using **Matplotlib** & **Seaborn**.
- Analyzed correlations between features and target variable.

4️⃣ **Model Building**
- Implemented multiple classification models:
  - **Logistic Regression**
  - **Random Forest**
  - **Decision Tree**
- Used **GridSearchCV** for hyperparameter tuning.

5️⃣ **Model Evaluation**
- Evaluated using metrics:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-Score**
- Plotted **Confusion Matrix**.

---

## 📈 Key Insights:
✔ Applicants with **good credit history** have a significantly higher approval chance.  
✔ **Applicant Income** and **Loan Amount** influence approval but credit history is the most important factor.  

---

## 🛠 Tools & Technologies:
- **Python**
- **Libraries**: `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`, `Scikit-learn`
- **Jupyter Notebook** for development.

---

