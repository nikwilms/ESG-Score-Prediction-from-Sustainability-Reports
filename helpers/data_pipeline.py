# Data Pipeline
def data_pipeline(df):
    # add new data in csv format

    # Stage 1: Text Preprocessing
    df["preprocessed_text"] = df["text"].apply(preprocess_text)

    # Stage 2: Adding spelling correction
    df["preprocessed_text"] = df["preprocessed_text"].apply(add_spelling_correction)

    # Stage 3: Adding ngrams
    df["preprocessed_text"] = df["preprocessed_text"].apply(add_ngrams)

    # Stage 4: Add to csv, if csv is not empty

    return df
