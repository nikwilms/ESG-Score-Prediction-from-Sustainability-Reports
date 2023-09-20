import ray
import spacy
import re
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
import os


def preprocess_text(df, checkpoint_file="preprocessing_checkpoint.parquet", checkpoint_interval=100):
    """
    This script defines a function for preprocessing text data using spaCy and Ray. It loads a spaCy model,
    preprocesses text in a given DataFrame, and saves the preprocessed data to a checkpoint and a CSV file.

    Parameters:
        - df (pandas.DataFrame): The input DataFrame containing the 'content'
          column with text data.
        - checkpoint_file (str): The name of the checkpoint file to save
          intermediate results. Default is 'preprocessing_checkpoint.parquet'.
        - checkpoint_interval (int): The number of records to process before
          saving a checkpoint. Default is 100.

    Returns:
        - pandas.DataFrame: A DataFrame containing the preprocessed text data in a new 'preprocessed_content' column.

    Usage Example:

        # Import the function
        from preprocessing import preprocess_text

        # Load or create your DataFrame
        df = pd.read_csv('your_data.csv')

        # Preprocess the text and get the preprocessed DataFrame
        preprocessed_df = preprocess_text(df)

        # Now, preprocessed_df contains the preprocessed text

    Note:
        - Make sure to have the 'en_core_web_sm' spaCy model installed. You can install it with 'python -m spacy download en_core_web_sm'.
        - Adjust 'checkpoint_file' and 'checkpoint_interval' as needed based on your project requirements.
        - The final preprocessed DataFrame is returned and can be used for downstream tasks.
    """

    # Load the spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Set a higher limit based on your text length
    nlp.max_length = 2000000

    # Define the preprocessing function
    def preprocess_with_spacy(text):
        # Remove hyphens followed by line breaks
        text = re.sub(r"-(?:\n|\r\n?)", " ", text)

        # Tokenize and process the text
        doc = nlp(text)

        # Use list comprehension for token filtering and processing
        preprocessed_tokens = [
            token.lemma_.lower()
            for token in doc
            if not (
                token.is_stop or token.is_punct
            )  # Filter out stop words and punctuation
            and token.pos_ in {"NOUN", "VERB", "ADJ"}  # Filter by POS tags
            and len(token.lemma_) > 2  # Remove short tokens
            and token.lemma_.isalpha()  # Remove non-alphanumeric tokens
        ]

        # Join the preprocessed tokens into a string
        preprocessed_text = " ".join(preprocessed_tokens)

        return preprocessed_text

    # Initialize Ray
    ray.init()

    # Check if checkpoint exists, if so, load it
    if os.path.exists(checkpoint_file):
        df = pd.read_parquet(checkpoint_file)
    else:
        # Preprocess and save checkpoint periodically
        df["preprocessed_content"] = None

        for i, text in enumerate(df["content"]):
            if i % checkpoint_interval == 0:
                df.to_parquet(checkpoint_file)

            preprocessed_text = preprocess_with_spacy(text)
            df.at[i, "preprocessed_content"] = preprocessed_text

        # Save the final dataframe to CSV
        df.to_csv("../data/preprocessed_data_text_format.csv", index=False)

        # Optionally, save the dataframe as a checkpoint
        df.to_parquet(checkpoint_file)

    # Return the preprocessed DataFrame
    return df
