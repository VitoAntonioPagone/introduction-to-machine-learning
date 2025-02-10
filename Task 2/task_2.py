import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer, enable_halving_search_cv 
from sklearn.impute import IterativeImputer
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.model_selection import HalvingGridSearchCV

X_train = pd.read_csv("train_features.csv")
y_train = pd.read_csv("train_labels.csv")
X_test = pd.read_csv("test_features.csv")

pid = pd.DataFrame(np.array(X_test.iloc[:, 0].drop_duplicates()))

pid_train = X_train.iloc[:, 0]
pid_test = X_test.iloc[:, 0]
X_train = X_train.drop(labels = ['pid', 'Time'], axis = 1)
X_test = X_test.drop(labels = ['pid', 'Time'], axis = 1)

scaler_train = preprocessing.StandardScaler().fit(X_train)
X_train = pd.DataFrame(scaler_train.transform(X_train))
scaler_test = preprocessing.StandardScaler().fit(X_test)
X_test = pd.DataFrame(scaler_test.transform(X_test))

X_train_mean = pd.concat([pid_train, X_train], axis = 1).groupby(axis = 0, by = "pid", sort = False).mean()
X_test_mean = pd.concat([pid_test, X_test], axis = 1).groupby(axis = 0, by = "pid", sort = False).mean()

X_train_mask = X_train_mean.notna().replace([True, False], [1, 0])
X_test_mask = X_test_mean.notna().replace([True, False], [1, 0])

imputer = IterativeImputer()
imputer.fit(X_train)
X_train = pd.DataFrame(imputer.transform(X_train))
imputer.fit(X_test)
X_test = pd.DataFrame(imputer.transform(X_test))

X_train_mean = pd.concat([pid_train, X_train], axis = 1).groupby(axis = 0, by = "pid", sort = False).mean()
X_test_mean = pd.concat([pid_test, X_test], axis = 1).groupby(axis = 0, by = "pid", sort = False).mean()

X_train = pd.concat([X_train_mean, X_train_mask], axis = 1)
X_test = pd.concat([X_test_mean, X_test_mask], axis = 1)

param_grid = {
    'estimator__max_depth': [10, 40, 70, 100, None],
    'estimator__max_features': ['auto', 'sqrt', 'log2'],
    'estimator__n_estimators': [200, 600, 1000, 1400, 1800]
}

multi_classifier = MultiOutputClassifier(RandomForestClassifier())
grid_search_classifier = HalvingGridSearchCV(multi_classifier, param_grid, cv = 3, scoring = "roc_auc", n_jobs = -1)
grid_search_classifier.fit(X_train, y_train.iloc[:, 1:12])
y_hat_classifier = grid_search_classifier.predict_proba(X_test)

multi_regressor = MultiOutputRegressor(RandomForestRegressor())
grid_search_regressor = HalvingGridSearchCV(multi_regressor, param_grid, cv = 3, scoring = "r2", n_jobs = -1)
grid_search_regressor.fit(X_train, y_train.iloc[:, 12:])
y_hat_regressor = grid_search_regressor.predict(X_test)

y_hat_classifier = pd.DataFrame(np.transpose(np.array(y_hat_classifier)[:, :,  1]))
y_hat_regressor = pd.DataFrame(y_hat_regressor)
y_hat = pd.concat([pid, y_hat_classifier, y_hat_regressor], axis = 1)
y_hat.columns = y_train.columns

y_hat.to_csv("results.csv", index = False, float_format = "%.3f", compression = "zip")
