from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import optuna
from math import sqrt
import pickle


def tune_nn_hyperparameters(X_train, y_train, X_test, y_test, n_trials=100):
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
        train_rmse = sqrt(-neg_mse.mean())

        model.fit(X_train, y_train)
        test_predictions = model.predict(X_test)
        test_rmse = sqrt(mean_squared_error(y_test, test_predictions))

        print(f"Training RMSE: {train_rmse}, Test RMSE: {test_rmse}")

        return test_rmse  # Objective is to minimize test RMSE

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print("Best hyperparameters:", study.best_params)
    print("Best Test RMSE:", study.best_value)

    with open("../models/Neural_Network/best_params_nn.pkl", "wb") as f:
        pickle.dump(study.best_params, f)

    return study.best_params
