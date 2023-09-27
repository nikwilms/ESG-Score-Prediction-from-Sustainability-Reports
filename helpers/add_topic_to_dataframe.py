import pandas as pd


def add_topic_to_dataframe(df, best_model, corpus):
    # Get topic distributions for each document
    topic_distributions = [
        best_model.get_document_topics(bow, minimum_probability=0) for bow in corpus
    ]

    # Convert to a format that can be added to the DataFrame
    topic_df = pd.DataFrame(
        [[topic_prob for _, topic_prob in doc] for doc in topic_distributions],
        columns=[f"Topic_{i}" for i in range(best_model.num_topics)],
    )

    # Add topic features to the original DataFrame
    df_with_topics = pd.concat(
        [df.reset_index(drop=True), topic_df.reset_index(drop=True)], axis=1
    )

    return df_with_topics


# Example usage:
"""
# Run the Optuna study to get the best model, corpus, and dictionary
best_model, corpus, dictionary = execute_optuna_study(df, n_trials=10)

# Add the topics to the DataFrame
df_with_topics = add_topic_to_dataframe(df, best_model, corpus)"""
