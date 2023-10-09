from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import optuna
import pickle
from math import sqrt


def tune_gradientboosting_hyperparameters(
    X_train, y_train, X_test, y_test, n_trials=100
):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["auto", "sqrt"]),
            "loss": "huber",
        }

        model = GradientBoostingRegressor(**params)
        kf = KFold(n_splits=5, shuffle=True, random_state=3)
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

    with open("../models/GradientBoosting/best_params_gradientboosting.pkl", "wb") as f:
        pickle.dump(study.best_params, f)

    return study.best_params
