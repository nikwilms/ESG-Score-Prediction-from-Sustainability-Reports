from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
import optuna


def train_and_evaluate_model(X_train, y_train, X_val, y_val, best_params):
    """
    Train and evaluate an XGBRegressor model using best hyperparameters.

    Parameters:
    - X_train: Feature matrix for training
    - y_train: Target vector for training
    - X_val: Feature matrix for validation
    - y_val: Target vector for validation
    - best_params: Dictionary of best hyperparameters

    Returns:
    - final_model: Trained XGBRegressor model
    - mse: Mean squared error on the validation set
    """

    final_model = XGBRegressor(**best_params, n_jobs=n_jobs)

    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)

    print(f"Mean Squared Error: {mse}")
    return final_model, mse


# How to use this function:
# final_model, mse = train_and_evaluate_model(X_train, y_train, X_val, y_val, best_params)
