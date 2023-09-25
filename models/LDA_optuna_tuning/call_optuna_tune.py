import pyLDAvis.gensim_models as gensimvis
import optuna.visualization as vis
import optuna
import pyLDAvis
from gensim import corpora
import logging

# import functions
from models.LDA_optuna_tuning.tune_lda_optuna import (
    train_lda,
    objective,
)


def preprocess_data(df):
    # Tokenize the 'preprocessed_content' column
    tokenized_data = df["preprocessed_content"].apply(lambda x: x.split())

    # DEBUGGING
    # print("Sample tokenized data:", tokenized_data[:2])

    # Create a Gensim dictionary from the tokenized data
    dictionary = corpora.Dictionary(tokenized_data)

    # Filter extremes
    # dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=None)

    # Create the corpus
    corpus = [dictionary.doc2bow(text) for text in tokenized_data]

    # DEBUGGING
    # print("Sample corpus data:", corpus[:2])
    # print("Dictionary:", list(dictionary.items())[:10])

    return corpus, dictionary


def execute_optuna_study(df, n_trials=10):
    corpus, dictionary = preprocess_data(df)

    study = optuna.create_study(direction="maximize")

    # DEBUGGING
    print(f"First element of corpus: {corpus[0]}")
    print(f"Length of dictionary: {len(dictionary)}")
    study.optimize(lambda trial: objective(trial, corpus, dictionary), n_trials=5)

    """ study.optimize(
        lambda trial: objective(trial, corpus, dictionary), n_trials=n_trials
    )"""

    logger = logging.getLogger(__name__)
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best trial params: {study.best_trial.params}")

    # Retrieve the best model
    best_params = study.best_trial.params
    best_model = train_lda(corpus, dictionary, **best_params)

    # Create pyLDAvis visualization
    lda_display = gensimvis.prepare(best_model, corpus, dictionary)
    pyLDAvis.show(lda_display)  # This will open a web browser

    # Create contour plot
    contour_plot = vis.plot_contour(study)
    contour_plot.show()

    # Create parameter importances plot
    param_importances_plot = vis.plot_param_importances(study)
    param_importances_plot.show()
