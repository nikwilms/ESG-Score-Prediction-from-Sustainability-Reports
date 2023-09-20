import ray
import spacy
import re
from spacy.lang.en.stop_words import STOP_WORDS
import os
import pandas as pd

def preprocess_text_data(df, content_column_name='content', checkpoint_interval=100, checkpoint_file="preprocessing_checkpoint.parquet", max_length=2000000):
    """
    Preprocess a DataFrame containing text data using spaCy.

    Args:
        df (pd.DataFrame): The DataFrame containing the text data.
        content_column_name (str): The name of the column in df that contains the text data.
        checkpoint_interval (int): The interval at which to checkpoint the preprocessing progress.
        checkpoint_file (str): The name of the checkpoint file for saving/loading the DataFrame.
        max_length (int): The maximum length allowed for spaCy text processing.

    Returns:
        pd.DataFrame: The DataFrame with an additional 'preprocessed_content' column containing the preprocessed text.

    Example:
        df = pd.read_csv("your_text_data.csv")
        df = preprocess_text_data(df)
        df.to_csv("preprocessed_text_data.csv", index=False)
    """

    # Load the spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Set a higher limit based on your text length
    nlp.max_length = max_length

    # Check if checkpoint exists, if so, load it
    if os.path.exists(checkpoint_file):
        df = pd.read_parquet(checkpoint_file)
    else:
        # Preprocess and save checkpoint periodically
        df['preprocessed_content'] = None

        for i, text in enumerate(df[content_column_name]):
            if i % checkpoint_interval == 0:
                df.to_parquet(checkpoint_file)

            preprocessed_text = preprocess_with_spacy(text, nlp)
            df.at[i, 'preprocessed_content'] = preprocessed_text

        # Save the final dataframe to CSV
        df.to_csv('preprocessed_data_text_format.csv', index=False)

        # Optionally, save the dataframe as a checkpoint
        df.to_parquet(checkpoint_file)

    return df

# Usage example:
# df = pd.read_csv("your_text_data.csv")
# df = preprocess_text_data(df)
# df.to_csv("preprocessed_text_data.csv", index=False)
