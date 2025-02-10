# Imputation on Medical Time Series Followed by Classification and Regression Task  

## Overview  
This project focuses on handling missing values in medical time-series data using imputation techniques, followed by classification and regression tasks. The goal is to predict medical test orders, sepsis events, and vital signs evolution for ICU patients.  

## Features  
- **Data Preprocessing:** Handles missing values using `IterativeImputer`.  
- **Feature Scaling:** Standardizes features using `StandardScaler`.  
- **Classification Task:** Predicts binary outcomes (e.g., medical test orders, sepsis events).  
- **Regression Task:** Predicts continuous values (e.g., mean vital signs).  
- **Model Optimization:** Uses `HalvingGridSearchCV` to tune hyperparameters for `RandomForestClassifier` and `RandomForestRegressor`.  

## Dependencies  
Ensure you have the following libraries installed:  
```bash
pip install pandas numpy scikit-learn
