from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from math import sqrt
import optuna
import pickle


def optimize_stacking(X, y, n_trials=50):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    def objective(trial):
        rf_n_estimators = trial.suggest_int("rf_n_estimators", 50, 200)
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        xgb_n_estimators = 1000
        xgb_max_depth = trial.suggest_int("xgb_max_depth", 2, 32, log=True)
        xgb_learning_rate = trial.suggest_float(
            "xgb_learning_rate", 1e-3, 0.1, log=True
        )
        lasso_alpha = trial.suggest_float("lasso_alpha", 1e-4, 1e-1, log=True)

        rf = RandomForestRegressor(n_estimators=rf_n_estimators, max_depth=rf_max_depth)
        xgb = XGBRegressor(
            n_estimators=xgb_n_estimators,
            max_depth=xgb_max_depth,
            learning_rate=xgb_learning_rate,
        )
        meta_model = Lasso(alpha=lasso_alpha)

        stacking_regressor = StackingRegressor(
            estimators=[("rf", rf), ("xgb", xgb)],
            final_estimator=meta_model,
        )

        kf = KFold(n_splits=5)
        neg_mse = cross_val_score(
            stacking_regressor,
            X_train,
            y_train,
            cv=kf,
            scoring="neg_mean_squared_error",
        )
        rmse = sqrt(-neg_mse.mean())

        stacking_regressor.fit(X_train, y_train)
        y_pred = stacking_regressor.predict(X_test)
        test_rmse = sqrt(mean_squared_error(y_test, y_pred))

        print(f"Train RMSE: {rmse}, Test RMSE: {test_rmse}")

        return test_rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    with open("../models/best_params_ensemble.pkl", "wb") as f:
        pickle.dump(study.best_params, f)

    return study.best_params


# Usage example:
# best_params = optimize_stacking(X, y, n_trials=50)
