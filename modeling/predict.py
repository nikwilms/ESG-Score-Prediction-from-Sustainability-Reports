import sys
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error
import warnings
import mlflow
from mlflow.sklearn import load_model

warnings.filterwarnings("ignore")

from feature_engineering import (
    fill_missing_values,
    drop_column,
    transform_altitude,
)

print("Number of arguments:", len(sys.argv), "arguments.")
print("Argument List:", str(sys.argv))

# in an ideal world this would validated
model_path = sys.argv[1]
X_test_path = sys.argv[2]
y_test_path = sys.argv[3]

# load the model from disk
# model_path = "models/linear"
loaded_model = load_model(model_path)
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path)

# feature eng on test data
print("Feature engineering")
X_test = transform_altitude(X_test)
X_test = drop_column(X_test, col_name="Unnamed: 0.1")
X_test = drop_column(X_test, col_name="Quakers")
X_test = fill_missing_values(X_test)

y_test_pred = loaded_model.predict(X_test)
mse_test = mean_squared_error(y_test, y_test_pred)
print(f"MSE on test is: {mse_test}")
