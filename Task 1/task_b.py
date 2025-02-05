import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

train_set = pd.read_csv("train.csv")
linear_X = train_set.drop(["Id", "y"], 1)
y = train_set["y"]

quadratic_X = linear_X.apply(np.square)
exponential_X = linear_X.apply(np.exp)
cosine_X = linear_X.apply(np.cos)
constant_X = pd.Series(np.ones((linear_X.shape[0],)))

X = pd.concat([linear_X, quadratic_X, exponential_X, cosine_X, constant_X], 1)

model = RidgeCV((0.1, 1, 10, 100, 200), cv=10, fit_intercept=False)
model.fit(X, y)

pd.DataFrame(model.coef_).to_csv("results.csv", index=False, header=False)
