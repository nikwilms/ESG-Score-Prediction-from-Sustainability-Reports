from helpers.topic_modelling import generate_tfidf
from helpers.topic_modelling import generate_dtm
from models.LDA_optuna_tuning.tune_lda_optuna import objective
from models.LDA_optuna_tuning.call_optuna_tune import call_optuna_tune
from models.NMF.perform_NMF import perform_nmf


def topic_modelling_pipeline(df):

# Stage 1 - Generate TF-IDF matrix

    tfidf_matrix = generate_tfidf(df)

# Stage 2 - Generate DTM matrix

    dtm_matrix = generate_dtm(df)

# Stage 3 - Call optuna tune

    optuna_tune = call_optuna_tune()

#  Stage 4.1 - Call NMF (with tfidf)

    W_tfidf, H_tfid, feature_names_tfidf = perform_nmf(tfidf_matrix, n_topics=5, n_top_words=10)

# Stage 4.2 - Call NMF (with dtm)

    W_dtm, H_dtm, feature_names_dtm = perform_nmf(dtm_matrix, n_topics=5, n_top_words=10)

return optuna_tune, W_tfidf, H_tfid, feature_names_tfidf, W_dtm, H_dtm, feature_names_dtm
