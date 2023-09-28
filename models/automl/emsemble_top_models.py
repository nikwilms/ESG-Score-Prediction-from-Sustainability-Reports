from sklearn.base import clone
import numpy as np
import pandas as pd


def ensemble_top_models(
    pareto_front, X_train, y_train, X_test, metric="some_metric", top_n=5
):
    """
    Ensemble top models from TPOT's Pareto front.

    Parameters:
    - pareto_front: DataFrame containing models from TPOT's Pareto front
    - X_train, y_train: training data
    - X_test: test features
    - metric: the metric by which to sort the models
    - top_n: number of top models to ensemble

    Returns:
    - ensemble_predictions: ensembled predictions on the test set
    """
    # Sort models by the given metric
    top_models = pareto_front.sort_values(by=metric, ascending=False).head(top_n)

    # 1. Extract pipelines
    pipelines = [eval(model_code) for model_code in top_models["Pipeline"]]

    # 2. Fit pipelines
    fitted_pipelines = []
    for pipeline in pipelines:
        # Clone the pipeline to ensure that the original pipeline object is not modified
        cloned_pipeline = clone(pipeline)
        cloned_pipeline.fit(X_train, y_train)
        fitted_pipelines.append(cloned_pipeline)

    # 3. Make predictions
    predictions = []
    for fitted_pipeline in fitted_pipelines:
        y_pred = fitted_pipeline.predict(X_test)
        predictions.append(y_pred)

    # 4. Ensemble predictions (here using a simple average)
    ensemble_predictions = np.mean(predictions, axis=0)

    return ensemble_predictions


# Example usage (assuming pareto_front is a DataFrame containing your top models)
# ensemble_predictions = ensemble_top_models(pareto_front, X_train, y_train, X_test)
