from sklearn.model_selection import KFold
import optuna
import gensim
from gensim.models import CoherenceModel
import logging
import numpy as np
import warnings


# Placeholder function for LDA training
def train_lda(corpus, id2word, num_topics, alpha, eta):
    model = gensim.models.LdaMulticore(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        alpha=alpha,
        eta=eta,
        random_state=100,
        passes=5,
    )
    return model


# Coherence computation
def compute_coherence(model, corpus, dictionary):
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try:
            coherence_model = CoherenceModel(
                model=model, texts=corpus, dictionary=dictionary, coherence="c_v"
            )
            coherence = coherence_model.get_coherence()
        except RuntimeWarning as e:
            print(f"RuntimeWarning caught: {e}")
            print("Dumping all local variables for debugging:")
            locals_copy = locals().copy()
            for k, v in locals_copy.items():
                print(f"{k}: {v}")
            coherence = None

    return coherence


# K-Fold Cross-Validation
def cross_val_coherence(corpus, dictionary, num_topics, alpha, eta, k=5):
    kf = KFold(n_splits=k)
    avg_coherence = 0.0

    for train_idx, test_idx in kf.split(corpus):
        train_corpus = [corpus[i] for i in train_idx]
        model = train_lda(train_corpus, dictionary, num_topics, alpha, eta)
        avg_coherence += compute_coherence(model, train_corpus, dictionary)

    return avg_coherence / k


# Optuna objective function
def objective(trial, corpus, dictionary):
    alpha = trial.suggest_float("alpha", 0.01, 1)
    eta = trial.suggest_float("eta", 0.01, 1)
    num_topics = trial.suggest_int("num_topics", 10, 50)

    # Check if asymmetric priors are being used
    if isinstance(alpha, list) or isinstance(eta, list):
        print("Using asymmetric priors for alpha or eta.")

    # Before training LDA model
    if not corpus or not dictionary:
        print("Corpus or Dictionary is empty or None.")
        trial.report(float("nan"), step=0)
        trial.set_user_attr("fail_cause", "empty_corpus_or_dict")
        raise optuna.TrialPruned()

    model = train_lda(corpus, dictionary, num_topics, alpha, eta)

    # Before computing coherence
    if model is None:
        print("Model training failed.")
        trial.report(float("nan"), step=0)
        trial.set_user_attr("fail_cause", "model_training_failed")
        raise optuna.TrialPruned()

    # Check token counts for each topic
    for topic_id in range(num_topics):
        print(
            f"Top terms for topic {topic_id}: {model.get_topic_terms(topic_id, topn=10)}"
        )

    coherence_score = compute_coherence(model, corpus, dictionary)

    # In your objective function
    if not coherence_score or np.isnan(coherence_score) or np.isinf(coherence_score):
        print(
            f"Invalid coherence_score: {coherence_score} for alpha={alpha}, eta={eta}, num_topics={num_topics}"
        )
        trial.report(float("nan"), step=0)
        trial.set_user_attr("fail_cause", "nan_coherence")
        raise optuna.TrialPruned()

    return coherence_score
