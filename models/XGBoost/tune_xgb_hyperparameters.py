from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
import optuna


def tune_xgb_hyperparameters(X_train, y_train, n_trials=100, n_jobs=6):
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
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "n_jobs": n_jobs,
        }

        model = XGBRegressor(**params)

        # Using 5-Fold cross-validation to compute mean squared error
        kf = KFold(n_splits=5)
        mse = -cross_val_score(
            model, X_train, y_train, cv=kf, scoring="neg_mean_squared_error"
        ).mean()

        return mse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params


# How to use this function:
# best_params = tune_xgb_hyperparameters(X_train, y_train, n_trials=50)
