from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
import optuna
import pandas as pd  # For creating a DataFrame to display feature importances


def train_and_evaluate_model(X_train, y_train, X_val, y_val, best_params, n_jobs=6):
    """
    Train and evaluate an XGBRegressor model using best hyperparameters.

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

    final_model = XGBRegressor(**best_params, n_jobs=n_jobs)
    final_model.fit(X_train, y_train)

    # Extract feature importances
    feature_importances = final_model.feature_importances_

    # Create a DataFrame for feature importances
    feature_importance_df = pd.DataFrame(
        {"Feature": X_train.columns, "Importance": feature_importances}
    ).sort_values(by="Importance", ascending=False)

    print("Feature Importances:")
    print(feature_importance_df)

    y_pred = final_model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)

    print(f"RMSE: {rmse}")

    return final_model, rmse, feature_importance_df


# How to use this function:
# trained_model, validation_rmse, feature_importances_df = train_and_evaluate_model(X_train, y_train, X_val, y_val, best_params)
