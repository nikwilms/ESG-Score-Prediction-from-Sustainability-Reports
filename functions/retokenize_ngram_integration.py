def preprocess_with_spacy(text, bigram_model=None, trigram_model=None):
    """
    Preprocess the given text with spaCy and integrate bigrams and trigrams.
    Args:
    - text (str): Input text.
    - bigram_model (gensim.models.phrases.Phraser, optional): Bigram model. Defaults to None.
    - trigram_model (gensim.models.phrases.Phraser, optional): Trigram model. Defaults to None.
    Returns:
    - str: Preprocessed text.
    """
    # Remove hyphens followed by line breaks
    text = re.sub(r'-(?:\n|\r\n?)', ' ', text)
    
    # Tokenize and process the text
    doc = nlp(text)
    
    # Extract tokens
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    
    # Apply bigram and trigram models, if provided
    if bigram_model:
        tokens = bigram_model[tokens]
    if trigram_model:
        tokens = trigram_model[tokens]
    
    # Further preprocessing
    preprocessed_tokens = []
    for token in tokens:
        # Process the token using spaCy
        token_doc = nlp(token)
        # Take the first token from the processed doc (there should only be one)
        token = token_doc[0]
        
        # Check the POS tag of the token (e.g., noun, verb, adjective)
        # You can adjust this condition based on your requirements
        if token.pos_ in {"NOUN", "VERB", "ADJ"}:
            # Lemmatize the token and convert to lowercase
            lemma = token.lemma_.lower()

            # Remove short tokens and non-alphanumeric tokens
            if len(lemma) > 2 and lemma.isalpha():
                preprocessed_tokens.append(lemma)

    # Join the preprocessed tokens into a string
    preprocessed_text = " ".join(preprocessed_tokens)
    
    return preprocessed_text

# First, tokenize the entire dataset to detect n-grams
tokenized_reports = [text.split() for text in df['content']]
bigram_model, trigram_model = detect_ngrams(tokenized_reports)

# Now you can preprocess each report with n-gram integration
preprocessed_spacy = preprocess_with_spacy(text_snippet, bigram_model, trigram_model)
print(preprocessed_spacy)