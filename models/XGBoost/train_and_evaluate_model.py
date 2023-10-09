from xgboost import XGBRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd  # For creating a DataFrame to display feature importances


def train_and_evaluate_model(X_train, y_train, X_val, y_val, best_params, n_jobs=6):
    """
    Train and evaluate an XGBoost model using best hyperparameters.

    Parameters:
    - X_train: Feature matrix for training
    - y_train: Target vector for training
    - X_val: Feature matrix for validation
    - y_val: Target vector for validation
    - best_params: Dictionary of best hyperparameters
    - n_jobs: Number of parallel jobs (default is 6)

    Returns:
    - final_model: Trained XGBRegressor model
    - rmse: Root Mean Square Error on the validation set
    - feature_importance_df: DataFrame containing feature importances
    """
    best_params_with_subsampling_and_reg = {
        **best_params,
        "reg_alpha": 0.1,
        "reg_lambda": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "colsample_bylevel": 0.8,
        "colsample_bynode": 0.8,
    }

    final_model = XGBRegressor(**best_params_with_subsampling_and_reg, n_jobs=n_jobs)
    final_model.fit(X_train, y_train)

    # Apply bagging to the model
    bagging_model = BaggingRegressor(
        base_estimator=final_model, n_estimators=10, random_state=0
    )
    bagging_model.fit(X_train, y_train)

    # Evaluate on the train set
    y_pred_train = bagging_model.predict(X_train)
    rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
    print(f"RMSE on Train set: {rmse_train}")

    # Extract feature importances
    feature_importances = final_model.feature_importances_

    # Create a DataFrame for feature importances
    feature_importance_df = pd.DataFrame(
        {"Feature": X_train.columns, "Importance": feature_importances}
    ).sort_values(by="Importance", ascending=False)

    print("Feature Importances:")
    print(feature_importance_df)

    # Evaluate on the validation set
    y_pred_val = bagging_model.predict(X_val)
    rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)
    print(f"RMSE on Validation set: {rmse_val}")

    return bagging_model, rmse_val, feature_importance_df
