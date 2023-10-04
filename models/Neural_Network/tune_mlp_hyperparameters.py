from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, KFold
import optuna
from math import sqrt


def tune_mlp_hyperparameters(X_train, y_train, n_trials=100):
    def objective(trial):
        params = {
            "hidden_layer_sizes": trial.suggest_categorical(
                "hidden_layer_sizes", [(50,), (100,), (50, 50)]
            ),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha": trial.suggest_float("alpha", 1e-4, 1e-1, log=True),
        }
        model = MLPRegressor(**params, max_iter=500)
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
