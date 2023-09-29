from sklearn.model_selection import train_test_split, KFold

import tpot2
import sklearn
import numpy as np
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("../data/model_data/model_data.csv")
    df_sample = df.sample(100, random_state=100)

    # columns to drop
    columns_to_drop = [
        "e_score",
        "s_score",
        "g_score",
        "unnamed: 0",
        "filename",
        "ticker",
        "year",
        "preprocessed_content",
        "ner_entities",
        "company_symbol",
        "total_score",
    ]

    # Separate features and target
    y = df["total_score"]
    X = df.drop(columns=columns_to_drop)

    # drop the last two rows
    X = X.iloc[:-2, :]
    y = y.iloc[:-2]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=100
    )

    scorer = sklearn.metrics.get_scorer("neg_mean_squared_error")

    # Initialize TPOT2 regressor with K-Fold cross-validation
    est = tpot2.TPOTEstimatorSteadyState(
        n_jobs=7,
        cv=KFold(n_splits=5),  # 5-Fold cross-validation
        verbose=2,
        classification=False,
        scorers=[scorer],
        scorers_weights=[1],
        max_eval_time_seconds=60 * 15,
        max_time_seconds=60 * 90,
    )

    # Fit the model
    est.fit(X_train, y_train)

    df_individuals = est.evaluated_individuals

    # Convert the 'mean_squared_error' column to numeric, errors='coerce' will replace non-numeric with NaN
    df_individuals["mean_squared_error"] = pd.to_numeric(
        df_individuals["mean_squared_error"], errors="coerce"
    )

    # Drop NaN values
    filtered_df = df_individuals.dropna(subset=["mean_squared_error"])

    # Sort the DataFrame by 'mean_squared_error' and get the top 10
    top_10_mse = filtered_df.nlargest(10, "mean_squared_error")

    print(top_10_mse)
