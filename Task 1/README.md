# Ridge Regression with K-Fold cross validation

## Overview Task 1a
This project implements a **model selection algorithm** using **10-fold cross-validation** with **scikit-learn's KFolds**. The goal is to determine the optimal **regularization parameter (α)** for **Ridge Regression** by evaluating model performance across multiple folds.

## Methodology
1. **Cross-Validation Setup**  
   - Used **KFolds cross-validator** (`k=10`) from `sklearn.model_selection.KFold`.
   - Enabled **shuffling** to improve generalization and reduce overfitting.

2. **Training and Validation**  
   - For each **regularization parameter (α)** in the outer loop:
     - Trained **10 models** (one per fold).
     - Each model was trained using **Ridge Regression** with `Ridge(alpha=lm)`.
     - The cross-validator split the data into **training and validation sets**, ensuring different subsets for each fold.
     - Predictions were made using `reg.predict()` on the validation set.

3. **Evaluation**  
   - Computed **Root Mean Squared Error (RMSE)** for each model.
   - Stored RMSE values in an array and calculated the **average RMSE** across all 10 folds.
   - The **optimal regularization parameter** was selected based on the **lowest average RMSE**.

   ## Overview Task 1b
   This project explores **nonlinear feature transformations** to improve model generalization, followed by **ridge regression** for predictive modeling. The goal is to optimize the **regularization parameter (λ)** and evaluate model accuracy.

## Methodology
1. **Feature Transformation**  
   - Applied **nonlinear transformations** to enhance feature representation.

2. **Model Training with Ridge Regression**  
   - Used **ridge regression** to train the model.
   - Optimized **regularization parameter (λ)** using **10-fold cross-validation**.
   - Tested **100 equally spaced λ values** from **1 to 100**.
   - **Best λ = 40**, achieving the lowest error.

3. **Performance Evaluation**  
   - **Root Mean Squared Error (RMSE)** was used to assess prediction accuracy.

4. **Comparison with Lasso Regression**  
   - Considered **Lasso regression** to reduce the number of features.
   - Lasso resulted in **higher errors** than Ridge Regression.
   - Decided to proceed with **ridge regression** due to better performance.

## Results
- **Optimal λ: 40** (selected based on lowest RMSE).
- **Ridge regression outperformed Lasso**, leading to better generalization.

