from helpers.topic_modelling.generate_dtm import create_dtm
from helpers.topic_modelling.generate_tfidf import create_tfidf
from helpers.merge_dataframes import merge_dataframes
from models.LDA_optuna_tuning.tune_lda_optuna import objective
from models.LDA_optuna_tuning.call_optuna_tune import preprocess_data
from models.LDA_optuna_tuning.call_optuna_tune import execute_optuna_study
from models.NMF.perform_NMF import perform_nmf
from helpers.add_topic_to_dataframe import add_topic_to_dataframe
from helpers.topic_modelling.get_embeddings import get_embeddings
from sklearn.decomposition import TruncatedSVD

import pandas as pd


def topic_modelling_pipeline(df, trials):

# Stage 1 - Run the Optuna study to get the best model, corpus, and dictionary
    best_model, corpus, dictionary = execute_optuna_study(df, n_trials=trials)

# Stage 2 - Add the topics to the DataFrame
    df_with_topics = add_topic_to_dataframe(df, best_model, corpus)

# Stage 3 - run TFIDF and add vectors to dataframe

    tfidf_matrix = create_tfidf(df_with_topics)
    df_with_topics = pd.concat([df_with_topics, tfidf_matrix], axis=1)

# Stage 4 Dimensionality reduction (Truncated SVD) for TFIDF vectors

    n_components_tfidf = 200  # You can adjust this based on your needs
    svd_tfidf = TruncatedSVD(n_components=n_components_tfidf)
    reduced_tfidf = svd_tfidf.fit_transform(tfidf_matrix)
    reduced_tfidf_df = pd.DataFrame(reduced_tfidf, columns=[f"tfidf_svd_dim_{i}" for i in range(n_components_tfidf)], index=df_with_topics.index)
    df_with_topics = pd.concat([df_with_topics, reduced_tfidf_df], axis=1)
    df_with_topics.drop(columns=tfidf_matrix.columns, inplace=True)

# Stage 5 - Word embedding with ESG-BERT

    df_with_topics['esg_bert_embeddings'] = df_with_topics['preprocessed_content'].apply(get_embeddings)
    bert_embeddings = pd.DataFrame(df_with_topics['esg_bert_embeddings'].tolist(), index=df_with_topics.index)
    df_with_topics = pd.concat([df_with_topics, bert_embeddings], axis=1)
    df_with_topics.drop(columns=['esg_bert_embeddings'], inplace=True)  # Drop the original embeddings column

# Stage 6 - Dimensionality reduction (Truncated SVD) for BERT embeddings

    # Number of components to r etain
    n_components = 200  # You can adjust this based on your needs
    svd = TruncatedSVD(n_components=n_components)
    reduced_embeddings = svd.fit_transform(bert_embeddings)
    reduced_embeddings_df = pd.DataFrame(reduced_embeddings, columns=[f"svd_dim_{i}" for i in range(n_components)], index=df_with_topics.index)
    df_with_topics = pd.concat([df_with_topics, reduced_embeddings_df], axis=1)

# Stage 7 Save the df_with_topics dataframe to CSV
    df_with_topics.to_csv('../data/ready_to_model/df_with_topics.csv')

# Stage 8 - merge all results into one dataframe

    df = pd.read_csv('../data/ready_to_model/df_with_topics.csv', index_col=0) # convert df_with_topics to df
    ESG_SP500 = pd.read_csv('../data/SP500_ESG_Score_average_per_year.csv', index_col=0) # convert ESG_SP500 to df
    merge_dataframes(df,ESG_SP500)

 
