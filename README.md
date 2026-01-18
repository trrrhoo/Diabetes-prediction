# Diabetes Prediction — Machine Learning Project

**Authors:** AMALAN KANNAN Théo, CHENENE Iliès, CLERE HAMELIN Maxime

## Goal
**Predict the diabetes risk score** and **classify the diabetes stage** for a patient using statistical learning methods based on lifestyle, demographic, and medical variables.

## Project Overview
This project explores two complementary tasks based on the [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/mohankrishnathalla/diabetes-health-indicators-dataset/data).

### 1. Regression — Predicting Diabetes Risk Score
We evaluate multiple regression models to understand how specific features influence the risk score.
* **Key aspects:**
    * Exploratory Data Analysis (EDA) & Correlation analysis.
    * Linear Regression & Assumptions verification.
    * Feature importance & confidence intervals.
    * Statistical inference to identify significant variables.

### 2. Classification — Predicting Diabetes Stage
A multi-class classification problem (5 classes: No Diabetes, Pre-Diabetes, Type 1, Type 2, Gestational) dealing with strong class imbalance.
* **Key techniques:**
    * **Models:** Random Forest Classifier, Gradient Boosting, Logistic Regression, SVM.
    * **Imbalance Handling:** Downsampling strategies and SMOTE (Synthetic Minority Over-sampling Technique).
    * **Evaluation:** Performance metrics beyond accuracy (Recall, Confusion Matrix) to minimize false negatives in a medical context.

## Main Results

### Regression
* **Performance:** High $R^2$ score achieved.
* **Insight:** A small subset of features explains most of the variance in the risk score.
* **Significance:** Statistically significant coefficients identified via p-values and confidence intervals, providing actionable insights on risk factors.

### Classification
* **Challenge:** Initial models were biased toward majority classes (Type 2 / No Diabetes).
* **Improvement:** After downsampling and tuning, we achieved a more balanced recall across stages.
* **Impact:** Significantly improved detection of rare and clinically important categories.

## Technologies Used
* **Python**
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (RandomForest, GradientBoosting, LinearRegression, PCA, etc.)
* **Imbalanced Learning:** `imblearn`
* **Visualization:** `matplotlib`, `seaborn`
* **Statistical Analysis:** `scipy`
