from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MaxAbsScaler
import optuna
import pickle
from math import sqrt


def tune_kernelridge_hyperparameters(X_train, y_train, X_test, y_test, n_trials=100):
    scaler = MaxAbsScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    def objective(trial):
        params = {
            "alpha": trial.suggest_float("alpha", 0.1, 1.0),
            "kernel": "polynomial",
            "degree": trial.suggest_int("degree", 1, 7),
        }
        model = KernelRidge(**params)
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

    with open("../models/Ridge/best_params_kernelridge.pkl", "wb") as f:
        pickle.dump(study.best_params, f)

    return study.best_params
