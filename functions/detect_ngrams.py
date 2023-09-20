import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import re
from gensim.models import Phrases, Phraser

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def detect_ngrams(docs):
    """
    Detect bigrams and trigrams using gensim's Phrases model.
    Args:
    - docs (list of lists): Tokenized documents.
    Returns:
    - bigram_model, trigram_model: Trained Phraser models for bigrams and trigrams.
    """
    # Detect bigrams
    bigrams = Phrases(docs, min_count=5, threshold=10)
    bigram_model = Phraser(bigrams)
    
    # Detect trigrams
    trigrams = Phrases(bigram_model[docs], threshold=10)
    trigram_model = Phraser(trigrams)
    
    return bigram_model, trigram_model