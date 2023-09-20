import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import re
from gensim.models.phrases import Phrases, Phraser
from spellchecker import SpellChecker

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def detect_ngrams_and_update_df(df): # df is the dataframe with the preprocessed_content column
    """
    Detect bigrams and trigrams using gensim's Phrases model, with integrated spell checking.
    Args:
    - df (pandas DataFrame): DataFrame containing the preprocessed_content column.
    Returns:
    - df: DataFrame with the 'content' column dropped.
    - bigram_model, trigram_model: Trained Phraser models for bigrams and trigrams.
    """
    # Extract tokenized sentences from 'preprocessed_content' column
    tokenized_sentences = df['preprocessed_content'].str.split().tolist()
    
    # Spell correction
    spell = SpellChecker()
    tokenized_sentences = [[spell.correction(word) for word in sentence if word is not None] for sentence in tokenized_sentences]
    
    # Remove None values
    tokenized_sentences = [[word for word in sentence if word is not None] for sentence in tokenized_sentences]
    
    # Detect bigrams
    bigrams = Phrases(tokenized_sentences, min_count=5, threshold=10)
    bigram_model = Phraser(bigrams)
    
    # Detect trigrams
    trigrams = Phrases(bigram_model[tokenized_sentences], threshold=10)
    trigram_model = Phraser(trigrams)
    
    # save models in new folder
    bigram_model.save("../../models/ngram_model")
    trigram_model.save("../../models/ngram_model")
    
    # Drop the 'content' column
    df = df.drop(columns=['content'])#,inplace=True
    
    return df, bigram_model, trigram_model


"""
   to load the model back:
    bigram_model = Phraser.load("path_to_save_bigram_model")
    trigram_model = Phraser.load("path_to_save_trigram_model")
"""
