from sklearn.ensemble import StackingRegressor, BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def perform_stacking(X, y, best_params_lasso, best_params_rf, best_params_xgb):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=100
    )

    # Initialize the models with the best hyperparameters
    lasso = Lasso(**best_params_lasso)

    # Combine default and regularization terms for Random Forest
    rf_params = {
        "max_depth": best_params_rf.get("max_depth", None),
        "min_samples_split": best_params_rf.get("min_samples_split", 2),
        "min_samples_leaf": best_params_rf.get("min_samples_leaf", 1),
        **best_params_rf,
    }

    # Combine default and regularization terms for XGBoost
    xgb_params = {
        "gamma": best_params_xgb.get("gamma", 0),
        "reg_alpha": best_params_xgb.get("alpha", 0),
        "reg_lambda": best_params_xgb.get("lambda", 1),
        **best_params_xgb,
    }

    # Initialize the models with the combined parameters
    rf = RandomForestRegressor(**rf_params)
    xgb = XGBRegressor(**xgb_params)

    # Apply bagging to the base models
    bagging_rf = BaggingRegressor(base_estimator=rf, n_estimators=10, random_state=0)
    bagging_xgb = BaggingRegressor(base_estimator=xgb, n_estimators=10, random_state=0)

    # Perform stacking
    stacking_regressor = StackingRegressor(
        estimators=[("bagging_rf", bagging_rf), ("bagging_xgb", bagging_xgb)],
        final_estimator=lasso,
    )

    # Fit the stacking regressor on the training data
    stacking_regressor.fit(X_train, y_train)

    # Evaluate the stacking model on the test set
    y_pred_test = stacking_regressor.predict(X_test)
    rmse_test = sqrt(mean_squared_error(y_test, y_pred_test))
    print(f"Root Mean Squared Error (RMSE) for test set: {rmse_test}")

    # Evaluate the stacking model on the training set
    y_pred_train = stacking_regressor.predict(X_train)
    rmse_train = sqrt(mean_squared_error(y_train, y_pred_train))
    print(f"Root Mean Squared Error (RMSE) for training set: {rmse_train}")

    return stacking_regressor, rmse_test


# Usage example
# best_params_lasso, best_params_rf, and best_params_xgb should be dictionaries containing the best hyperparameters for each model
# X and y are your feature matrix and target variable
# stacking_model, test_rmse = perform_stacking(X, y, best_params_lasso, best_params_rf, best_params_xgb)
