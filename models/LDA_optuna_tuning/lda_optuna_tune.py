import optuna
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models import CoherenceModel

# Objective function
def lda_optuna_tune(trial, df):
    # Prepare corpus and dictionary
    texts = df['preprocessed_content'].str.split().tolist()
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Define hyperparameters for tuning
    num_topics = trial.suggest_int('num_topics', 4, 20)
    alpha = trial.suggest_float('alpha', 0.01, 1, log=True)
    eta = trial.suggest_float('eta', 0.01, 1, log=True)

    # Create and train LDA model
    lda_model = LdaModel(corpus=corpus,
                         id2word=dictionary,
                         num_topics=num_topics,
                         random_state=42,
                         alpha=alpha,
                         eta=eta)

    # Evaluate LDA model
    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model_lda.get_coherence()

    return coherence_score

