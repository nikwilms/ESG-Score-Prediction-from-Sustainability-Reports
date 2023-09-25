from helpers.topic_modelling.generate_dtm import create_dtm
from helpers.topic_modelling.generate_tfidf import create_tfidf
from helpers.merge_dataframes import merge_dataframes
from models.LDA_optuna_tuning.tune_lda_optuna import objective
from models.LDA_optuna_tuning.call_optuna_tune import preprocess_data
from models.LDA_optuna_tuning.call_optuna_tune import execute_optuna_study
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

# Stage 5 - merge all results into one dataframe

    merge_dataframes(df,ESG_SP500)

# Stage 6 - save the final dataframe to CSV