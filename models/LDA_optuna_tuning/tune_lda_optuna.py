from sklearn.model_selection import KFold
import optuna
import gensim
from gensim.models.coherencemodel import CoherenceModel
import logging
import numpy as np


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
    coherence_model = CoherenceModel(
        model=model, texts=corpus, dictionary=dictionary, coherence="c_v"
    )
    coherence = coherence_model.get_coherence()

    # DEBUGGING
    # print(f"Coherence: {coherence}, Model Params: {model.alpha}, {model.eta}, {model.num_topics}")
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


def objective(trial, corpus, dictionary):
    print("Sample from corpus:", corpus[:1])
    print("Sample from dictionary:", list(dictionary.items())[:5])

    alpha = trial.suggest_float("alpha", 0.01, 1)
    eta = trial.suggest_float("eta", 0.01, 1)
    num_topics = trial.suggest_int("num_topics", 10, 50)

    print(
        f"Trying parameters: alpha={alpha}, eta={eta}, num_topics={num_topics}"
    )  # Debugging line

    model = train_lda(corpus, dictionary, num_topics, alpha, eta)
    coherence_score = compute_coherence(model, corpus, dictionary)

    if not coherence_score or np.isnan(coherence_score) or np.isinf(coherence_score):
        print(
            f"Invalid coherence_score: {coherence_score} for alpha={alpha}, eta={eta}, num_topics={num_topics}"
        )
        # raise optuna.exceptions.TrialPruned()

    return coherence_score
