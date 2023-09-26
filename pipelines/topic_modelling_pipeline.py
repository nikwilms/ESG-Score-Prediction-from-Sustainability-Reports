from helpers.topic_modelling.generate_dtm import create_dtm
from helpers.topic_modelling.generate_tfidf import create_tfidf
from helpers.merge_dataframes import merge_dataframes
from models.LDA_optuna_tuning.tune_lda_optuna import objective
from models.LDA_optuna_tuning.call_optuna_tune import preprocess_data
from models.LDA_optuna_tuning.call_optuna_tune import execute_optuna_study
from models.NMF.perform_NMF import perform_nmf
from helpers.add_topic_to_dataframe import add_topic_to_dataframe
import pandas as pd


def topic_modelling_pipeline(df, trials):

# Stage 1 - Generate TF-IDF matrix

    #tfidf_matrix = create_tfidf(df)

# Stage 2 - Generate DTM matrix

   # dtm_matrix = generate_dtm(df)

# Stage 3 - Call optuna tune

# Run the Optuna study to get the best model, corpus, and dictionary
    best_model, corpus, dictionary = execute_optuna_study(df, n_trials=trials)

# Add the topics to the DataFrame
    df_with_topics = add_topic_to_dataframe(df, best_model, corpus)

# Save the df_with_topics dataframe to CSV
    df_with_topics.to_csv('../data/ready_to_model_data/df_with_topics.csv')

#  Stage 4.1 - Call NMF (with tfidf)

   # W_tfidf, H_tfid, feature_names_tfidf = perform_nmf(tfidf_matrix, n_topics=5, n_top_words=10)

# Stage 4.2 - Call NMF (with dtm)

    #W_dtm, H_dtm, feature_names_dtm = perform_nmf(dtm_matrix, n_topics=5, n_top_words=10)

# Stage 5 - merge all results into one dataframe

    df = pd.read_csv('../data/ready_to_model_data/df_with_topics.csv', index_col=0) # convert df_with_topics to df
    ESG_SP500 = pd.read_csv('../data/SP500_ESG_Score_average_per_year.csv', index_col=0) # convert ESG_SP500 to df
    merge_dataframes(df,ESG_SP500)

# Stage 6 - save the final dataframe to CSV
 