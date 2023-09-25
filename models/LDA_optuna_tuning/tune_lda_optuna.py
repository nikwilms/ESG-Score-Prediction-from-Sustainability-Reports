from sklearn.model_selection import KFold
import optuna
import gensim
from gensim.models import CoherenceModel
import logging
import numpy as np
import warnings
import mlflow


# Note:
# - 'corpus' is a numerical representation used for training the LDA model.
# - 'tokenized_texts' contains the actual words and is used for computing coherence.
# Train the LDA model with 'corpus' but compute coherence using 'tokenized_texts'.


# Placeholder function for LDA training
def train_lda(corpus, id2word, num_topics, alpha, eta):
    """
    Train an LDA model on the given corpus.

    Args:
    - corpus (list): Bag-of-words representation of the documents.
    - id2word (Dictionary): Gensim dictionary mapping of id to word.
    - num_topics (int): Number of topics to be extracted from the training corpus.
    - alpha (float): Hyperparameter for LDA model.
    - eta (float): Hyperparameter for LDA model.

    Returns:
    - model (LdaMulticore): Trained LDA model.
    """

    model = gensim.models.LdaMulticore(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        alpha=alpha,
        eta=eta,
        random_state=100,
        passes=3,
        dtype=np.float64,
    )
    return model


def compute_coherence(model, tokenized_texts, dictionary):
    """
    Compute the coherence score of the given LDA model.

    Args:
    - model (LdaMulticore): Trained LDA model.
    - tokenized_texts (list): List of tokenized texts.
    - dictionary (Dictionary): Gensim dictionary mapping.

    Returns:
    - float: Coherence score.
    """

    coherence_model = CoherenceModel(
        model=model, texts=tokenized_texts, dictionary=dictionary, coherence="c_v"
    )
    coherence = coherence_model.get_coherence()
    return coherence


def cross_val_coherence(
    corpus, dictionary, tokenized_texts, num_topics, alpha, eta, k=5
):
    """
    Perform k-fold cross-validation on the given corpus and return the average coherence score.

    Args:
    - corpus (list): Bag-of-words representation of the documents.
    - dictionary (Dictionary): Gensim dictionary mapping of id to word.
    - tokenized_texts (list): List of tokenized texts.
    - num_topics (int): Number of topics to be extracted from the training corpus.
    - alpha (float): Hyperparameter for LDA model.
    - eta (float): Hyperparameter for LDA model.
    - k (int): Number of folds in k-fold cross-validation.

    Returns:
    - float: Average coherence score.
    """

    kf = KFold(n_splits=k)
    avg_coherence = 0.0

    for train_idx, _ in kf.split(corpus):  # Ignore test_idx
        train_corpus = [corpus[i] for i in train_idx]
        model = train_lda(train_corpus, dictionary, num_topics, alpha, eta)
        avg_coherence += compute_coherence(model, tokenized_texts, dictionary)

    return avg_coherence / k


# Optuna objective function
def objective(trial, corpus, dictionary, tokenized_texts):
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
    - trial (Trial): Optuna trial object.
    - corpus (list): Bag-of-words representation of the documents.
    - dictionary (Dictionary): Gensim dictionary mapping of id to word.
    - tokenized_texts (list): List of tokenized texts.

    Returns:
    - float: Coherence score of the trial.
    """
    with mlflow.start_run(run_name=f"Trial_{trial.number}", nested=True) as nested_run:
        alpha = trial.suggest_float("alpha", 0.01, 1)
        eta = trial.suggest_float("eta", 0.01, 1)
        num_topics = trial.suggest_int("num_topics", 10, 50)

        model = train_lda(corpus, dictionary, num_topics, alpha, eta)
        coherence_score = compute_coherence(model, tokenized_texts, dictionary)

        # Log parameters and metrics to the existing MLflow run
        mlflow.log_params(trial.params)
        mlflow.log_metric("coherence_score", coherence_score)

        return coherence_score
