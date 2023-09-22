import spacy
import re
import pandas as pd
import joblib

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000


def preprocess_with_spacy(text):
    """
    Preprocesses the input text using the Spacy library.

    Args:
        text (str): The input text to be preprocessed.

    Returns:
        tuple: A tuple containing the preprocessed text (str) and a list of named entities (list).
    """
    # Remove hyphens followed by line breaks
    text = re.sub(r"-(?:\n|\r\n?)", " ", text)

    # Tokenize and process the text
    doc = nlp(text)

    # Extract NER entities and store them as a list
    ner_entities = [ent.text for ent in doc.ents]

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

    return preprocessed_text, ner_entities


def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the text in the 'content' column of the input DataFrame using spaCy.

    Args:
        df (pd.DataFrame): Input DataFrame containing a 'content' column with text to preprocess.

    Returns:
        pd.DataFrame: Output DataFrame with preprocessed text in a new 'preprocessed_content' column and NER entities in a new 'ner_entities' column. The DataFrame is also saved to a CSV file at '../../data/preprocessed_text_with_ner.csv'.
    """
    # Initialize joblib for parallel processing
    num_cores = joblib.cpu_count()
    parallel = joblib.Parallel(n_jobs=num_cores, backend="multiprocessing")

    # Use joblib to parallelize the preprocessing step
    results = parallel(
        joblib.delayed(preprocess_with_spacy)(text) for text in df["content"]
    )

    # Unpack the results
    preprocessed_texts, ner_entities_list = zip(*results)

    # Add the preprocessed texts to the DataFrame
    df["preprocessed_content"] = preprocessed_texts

    # Add the NER entities as a new column
    df["ner_entities"] = ner_entities_list

    # Save the final dataframe to CSV
    df.to_csv("../../data/preprocessed_text_with_ner.csv", index=False)

    return df


# Example usage:
# preprocessed_df = preprocess_text(df)
