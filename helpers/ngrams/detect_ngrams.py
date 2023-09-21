import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import re
from gensim.models.phrases import Phrases, Phraser
from spellchecker import SpellChecker
import os

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def detect_ngrams(df, min_count=5, threshold=15): # df is the dataframe with the preprocessed_content column
    """
    Detect bigrams and trigrams using gensim's Phrases model.
    Args:
    - df (pandas DataFrame): DataFrame containing the 'preprocessed_content' column.
    - min_count (int): Ignore all words and bigrams with total collected count lower than this value.
    - threshold (float): Represent a score threshold for forming the phrases (higher means fewer phrases).
    Returns:
    - df: DataFrame with the 'content' column dropped.
    - bigram_model, trigram_model: Trained Phraser models for bigrams and trigrams.
    """
    # Extract tokenized sentences from 'preprocessed_content' column
    tokenized_sentences = df['preprocessed_content'].str.split().tolist()
    
    # Detect bigrams
    bigrams = Phrases(tokenized_sentences, min_count=min_count, threshold=threshold)
    bigram_model = Phraser(bigrams)
    
    # Detect trigrams
    trigrams = Phrases(bigram_model[tokenized_sentences], threshold=threshold)
    trigram_model = Phraser(trigrams)
    
    bigram_model.save('../data/ngrams/bigram_model')
    trigram_model.save('../data/ngrams/trigram_model')
    
    # Drop the 'content' column
    df.drop(columns=['content'], inplace=True)
    
    return df, bigram_model, trigram_model

# Example usage:
# Assuming df is your DataFrame containing 'content' and 'preprocessed_content' columns

# First, perform spell checking
# df = spell_check_df(df)

# Then, detect n-grams
# df, bigram_model, trigram_model = detect_ngrams(df)

"""
   to load the model back:
    bigram_model = Phraser.load("path_to_save_bigram_model")
    trigram_model = Phraser.load("path_to_save_trigram_model")
"""