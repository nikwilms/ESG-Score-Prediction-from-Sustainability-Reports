import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from math import sqrt


def perform_stacking(X_train, y_train, X_test, y_test):
    # Load pickled models
    with open("../models/ElasticNet/best_params_elasticnet.pkl", "rb") as f:
        elastic_net_params = pickle.load(f)

    with open(
        "../models/GradientBoostingReg/best_params_gradientboosting.pkl", "rb"
    ) as f:
        gradient_boosting_params = pickle.load(f)

    with open("../models/LightGBM/best_params_lightgbm.pkl", "rb") as f:
        lgbm_params = pickle.load(f)

    with open("../models/Random_Forest/best_params_rf.pkl", "rb") as f:
        random_forest_params = pickle.load(f)

    with open("../models/Ridge/best_params_kernelridge.pkl", "rb") as f:
        ridge_params = pickle.load(f)

    with open("../models/XGBoost/best_params_xgb_updated.pkl", "rb") as f:
        xgboost_params = pickle.load(f)

    with open("../models/Lasso/best_params_lasso.pkl", "rb") as f:
        lasso_params = pickle.load(f)

    # Initialize base models with loaded parameters
    base_models = [
        ("ElasticNet", ElasticNet(**elastic_net_params)),
        ("GradientBoosting", GradientBoostingRegressor(**gradient_boosting_params)),
        ("LGBM", lgb.LGBMRegressor(**lgbm_params)),
        ("RandomForest", RandomForestRegressor(**random_forest_params)),
        ("Ridge", KernelRidge(**ridge_params)),
        ("XGBoost", XGBRegressor(**xgboost_params)),
    ]

    # Initialize Lasso as the meta-model with loaded parameters
    lasso = Lasso(**lasso_params)

    # Apply Bagging to the base models
    bagging_models = [
        (name, BaggingRegressor(base_estimator=model, n_estimators=10, random_state=42))
        for name, model in base_models
    ]

    # Perform stacking
    stacking_regressor = StackingRegressor(
        estimators=bagging_models, final_estimator=lasso
    )

    # Fit the stacking regressor on the training data
    stacking_regressor.fit(X_train, y_train)

    # Save the model
    with open("stacking_regressor_model.pkl", "wb") as f:
        pickle.dump(stacking_regressor, f)

    # Evaluate the stacking model on the test set
    y_pred_test = stacking_regressor.predict(X_test)
    rmse_test = sqrt(mean_squared_error(y_test, y_pred_test))
    print(f"Root Mean Squared Error (RMSE) for test set: {rmse_test}")

    # Evaluate the stacking model on the training set
    y_pred_train = stacking_regressor.predict(X_train)
    rmse_train = sqrt(mean_squared_error(y_train, y_pred_train))
    print(f"Root Mean Squared Error (RMSE) for training set: {rmse_train}")

    # Save RMSE scores for future reference
    with open("rmse_scores.pkl", "wb") as f:
        pickle.dump({"Train": rmse_train, "Test": rmse_test}, f)

    # Plotting RMSE for a non-technical audience
    plt.figure(figsize=(10, 6))
    plt.bar(["Train Set", "Test Set"], [rmse_train, rmse_test], color=["blue", "green"])
    plt.title("Model Performance: RMSE Score")
    plt.ylabel("RMSE Score")
    plt.xlabel("Data Set")
    for i, v in enumerate([rmse_train, rmse_test]):
        plt.text(i, v + 0.01, str(round(v, 2)), ha="center")
    plt.show()


# Example usage
# perform_stacking(X_train, y_train, X_test, y_test)
