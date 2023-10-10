import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from math import sqrt
import optuna
import pickle


def tune_lightgbm_hyperparameters(X_train, y_train, X_test, y_test, n_trials=100):
    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "max_depth": trial.suggest_int("max_depth", 1, 25),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
            "n_estimators": 10000,
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1e-5, 1e-1, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 1.0, log=True),
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        }
        model = lgb.LGBMRegressor(**params)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
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
    print("Best Test RMSE:", study.best_value)

    with open("../models/LightGBM/best_params_lightgbm.pkl", "wb") as f:
        pickle.dump(study.best_params, f)

    return study.best_params
