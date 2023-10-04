from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
import optuna
from math import sqrt


def tune_lasso_hyperparameters(X_train, y_train, n_trials=100):
    def objective(trial):
        params = {
            "alpha": trial.suggest_float("alpha", 1e-4, 1.0, log=True),
        }
        model = Lasso(**params)
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
