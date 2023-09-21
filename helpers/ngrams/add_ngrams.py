from gensim.models.phrases import Phrases, Phraser

def retokenize_ngram_integration(text, bigram_model=None, trigram_model=None):
    """
    Integrate detected bigrams and trigrams into tokenized text.
    Args:
    - text (str): Input text.
    - bigram_model (gensim.models.phrases.Phraser, optional): Bigram model. Defaults to None.
    - trigram_model (gensim.models.phrases.Phraser, optional): Trigram model. Defaults to None.
    Returns:
    - str: Text with integrated bigrams and trigrams.
    """
    tokens = text.split()
    
    # Apply bigram and trigram models, if provided
    if bigram_model:
        tokens = bigram_model[tokens]
    if trigram_model:
        tokens = trigram_model[tokens]
    
    # Join the tokens back into a string
    retokenized_text = " ".join(tokens)
    
    return retokenized_text
