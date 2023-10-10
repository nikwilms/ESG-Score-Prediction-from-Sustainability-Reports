from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
import optuna
from math import sqrt
import pickle


def tune_xgb_hyperparameters(X_train, y_train, X_test, y_test, n_trials=100, n_jobs=6):
    """
    Tune hyperparameters for an XGBRegressor model using Optuna.

    Parameters:
    - X_train: Feature matrix for training
    - y_train: Target vector for training
    - n_trials: Number of trials for hyperparameter tuning (default is 100)

    Returns:
    - best_params: Dictionary of best hyperparameters
    """

    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "n_estimators": 1000,
            "verbosity": 0,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 20),
            "subsample": trial.suggest_float("subsample", 0.05, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "n_jobs": n_jobs,
            "random_state": 42,
        }

        model = XGBRegressor(**params)

        kf = KFold(n_splits=4, shuffle=True, random_state=42)
        neg_mse = cross_val_score(
            model, X_train, y_train, cv=kf, scoring="neg_mean_squared_error"
        )
        train_rmse = sqrt(-neg_mse.mean())

        model.fit(X_train, y_train)
        test_predictions = model.predict(X_test)
        test_rmse = sqrt(mean_squared_error(y_test, test_predictions))

        print(f"Training RMSE: {train_rmse}, Test RMSE: {test_rmse}")

        return test_rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("Best hyperparameters:", study.best_params)
    print("Best RMSE:", study.best_value)

    with open("../models/XGBoost/best_params_xgb_updated.pkl", "wb") as f:
        pickle.dump(study.best_params, f)

    return study.best_params
