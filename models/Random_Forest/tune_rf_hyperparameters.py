from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
import optuna
from math import sqrt


def tune_rf_hyperparameters(X_train, y_train, n_trials=100):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 4, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["auto", "sqrt", "log2"]
            ),
        }
        model = RandomForestRegressor(**params)
        kf = KFold(n_splits=5)
        neg_mse = cross_val_score(
            model, X_train, y_train, cv=kf, scoring="neg_mean_squared_error"
        )
        rmse = sqrt(-neg_mse.mean())
        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print("Best hyperparameters:", study.best_params)
    print("Best RMSE:", study.best_value)
    return study.best_params
