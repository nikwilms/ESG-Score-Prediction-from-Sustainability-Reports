import optuna
import pyLDAvis.gensim_models as gensimvis
import plotly
from gensim.models import LdaModel
import gensim
from models.LDA_optuna_tuning.tune_lda_optuna import objective


def call_optuna_tune(n_trials=50):
    """
    Tune hyperparameters for LDA model using Optuna.

    Args:
        df (DataFrame): DataFrame containing the 'preprocessed_content' column.
        n_trials (int): Number of trials for Optuna optimization. Default is 50.

    Returns:
        dict: Dictionary containing best parameters, best coherence score, and pyLDAvis visualization.
    """
    # Set the Optuna logging level to INFO
    optuna.logging.set_verbosity(optuna.logging.INFO)
    # Create a study object and specify the direction is 'maximize'
    study = optuna.create_study(direction="maximize")

    # Optimize the study
    study.optimize(objective, n_trials=n_trials)

    # Get the best parameters and best score
    best_params = study.best_params
    best_score = study.best_value

    corpus = study.best_trial.values[1]
    dictionary = study.best_trial.values[2]

    # Train the LDA model with the best parameters
    lda_model = gensim.models.LdaMulticore(
        workers=7,
        corpus=corpus,
        id2word=dictionary,
        num_topics=study.best_trial.params["num_topics"],
        alpha=study.best_trial.params["alpha"],
        eta=study.best_trial.params["eta"],
    )

    # Create pyLDAvis visualization
    lda_display = gensimvis.prepare(lda_model, corpus, dictionary)

    # Create contour plot
    contour_plot = optuna.visualization.plot_contour(study)

    # Create parameter importances plot
    param_importances_plot = optuna.visualization.plot_param_importances(study)

    return {
        "best_params": best_params,
        "best_score": best_score,
        "pyLDAvis": lda_display,
        "contour_plot": contour_plot,
        "param_importances_plot": param_importances_plot,
    }
