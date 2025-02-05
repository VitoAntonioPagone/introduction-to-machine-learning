import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

train_set = pd.read_csv("train.csv")
X = train_set.drop("y", 1)
y = train_set["y"]

model_1 = Ridge(0.1)
model_2 = Ridge(1)
model_3 = Ridge(10)
model_4 = Ridge(100)
model_5 = Ridge(200)

rmse = np.zeros((5, 10))

kf = KFold(10)
i = 0
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model_1.fit(X_train, y_train)
    model_2.fit(X_train, y_train)
    model_3.fit(X_train, y_train)
    model_4.fit(X_train, y_train)
    model_5.fit(X_train, y_train)
    rmse[0][i] = mean_squared_error(y_test, model_1.predict(X_test)) ** 0.5
    rmse[1][i] = mean_squared_error(y_test, model_2.predict(X_test)) ** 0.5
    rmse[2][i] = mean_squared_error(y_test, model_3.predict(X_test)) ** 0.5
    rmse[3][i] = mean_squared_error(y_test, model_4.predict(X_test)) ** 0.5
    rmse[4][i] = mean_squared_error(y_test, model_5.predict(X_test)) ** 0.5
    i = i + 1

mean_rmse = np.mean(rmse, 1)
pd.DataFrame(mean_rmse).to_csv("results.csv", index=False, header=False)
