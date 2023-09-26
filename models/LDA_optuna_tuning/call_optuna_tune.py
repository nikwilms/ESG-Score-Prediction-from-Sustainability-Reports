import pyLDAvis.gensim_models as gensimvis
import optuna.visualization as vis
import optuna
import pyLDAvis
from gensim import corpora
import logging
import mlflow
from optuna.pruners import MedianPruner


# import functions
from models.LDA_optuna_tuning.tune_lda_optuna import (
    train_lda,
    objective,
)


def preprocess_data(df):
    """
    Preprocess the DataFrame to produce a Gensim dictionary and corpus.

    Args:
    - df (DataFrame): DataFrame containing the 'preprocessed_content' column.

    Returns:
    - corpus (list): Bag-of-words representation of the documents.
    - dictionary (Dictionary): Gensim dictionary mapping of id to word.
    - tokenized_data (list): List of tokenized texts.
    """
    # Tokenize the 'preprocessed_content' column
    tokenized_data = df["preprocessed_content"].apply(lambda x: x.split())

    # Create a Gensim dictionary from the tokenized data
    dictionary = corpora.Dictionary(tokenized_data)

    # Filter out words that occur less than 10 documents, or more than 50% of the documents
    # dictionary.filter_extremes(no_below=10, no_above=0.5)

    # Create the corpus
    corpus = [dictionary.doc2bow(text) for text in tokenized_data]

    # Filter out empty documents
    corpus = [doc for doc in corpus if doc]

    print(len(corpus))
    print(corpus[:5])
    print(len(dictionary))

    return corpus, dictionary, tokenized_data


def execute_optuna_study(df, n_trials=10):
    """
    Execute Optuna study for hyperparameter tuning of the LDA model.

    Args:
    - df (DataFrame): DataFrame containing the 'preprocessed_content' column.
    - n_trials (int): Number of trials for Optuna optimization.

    Returns:
    - None: Function executes the Optuna study and generates visualizations.
    """
    # Start an MLflow run
    with mlflow.start_run(run_name="Optuna_Study") as parent_run:
        # Initialize MLflow Optuna store
        mlflow_storage = "sqlite:///../data/mlflow/mlflow.db"

        # Preprocessing data and getting corpus, dictionary, and tokenized texts
        corpus, dictionary, tokenized_texts = preprocess_data(df)

        # Configure the median pruner
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)

        # Creating an Optuna study object and specifying the direction is 'maximize'.
        study = optuna.create_study(
            direction="maximize",
            storage=mlflow_storage,
            pruner=pruner,
        )

        # Optimizing the study, the objective function is passed in as the first argument.
        study.optimize(
            lambda trial: objective(trial, corpus, dictionary, tokenized_texts),
            n_trials=n_trials,
        )

        # Retrieve best model parameters
        best_params = study.best_trial.params
        best_model = train_lda(corpus, dictionary, **best_params)

        # Log the best parameters to MLflow
        mlflow.log_params(study.best_trial.params)

        # Use MLflow's API to save the best model
        best_model_path = "../data/lda/best_lda_model"
        best_model.save(best_model_path)
        mlflow.log_artifact(best_model_path)

        # Create a pyLDAvis visualization
        lda_display = gensimvis.prepare(best_model, corpus, dictionary)
        # pyLDAvis.show(lda_display)

        # Save the pyLDAvis visualization to HTML
        lda_html_path = "../data/lda/lda.html"
        pyLDAvis.save_html(lda_display, lda_html_path)

        # Log the file to MLflow
        mlflow.log_artifact(lda_html_path)

        # Create a contour plot (assuming 'vis' is already imported and functional)
        contour_plot = vis.plot_contour(study)
        contour_plot.show()

        # Create a parameter importance plot
        param_importances_plot = vis.plot_param_importances(study)
        param_importances_plot.show()

        return best_model, corpus, dictionary
