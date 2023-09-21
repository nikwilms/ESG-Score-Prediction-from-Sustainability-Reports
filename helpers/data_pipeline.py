# Data Pipeline
def data_pipeline(df):
    # Stage 1: Text Preprocessing
    df["preprocessed_text"] = df["text"].apply(preprocess_text)

    # Stage 2: Adding spelling correction
    df["preprocessed_text"] = df["preprocessed_text"].apply(add_spelling_correction)
    
    # Stage 3: Adding ngrams
    df["preprocessed_text"] = df["preprocessed_text"].apply(add_ngrams)

    # Stage 3: Removing most common words
    # df["preprocessed_text"] = df["preprocessed_text"].apply(remove_most_common_words)

    # Stage 4: Removing least common words
    # df["preprocessed_text"] = df["preprocessed_text"].apply(remove_least_common_words)

    # Stage 5: Add to csv, if csv is not empty

    return df
