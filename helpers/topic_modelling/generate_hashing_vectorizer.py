from sklearn.feature_extraction.text import HashingVectorizer
import pandas as pd


def create_hashing_vectorizer(df, n_features=10000):
    """
    Generate a Hashing Vectorizer Matrix using the preprocessed content in the DataFrame.

    Args:
        df (DataFrame): DataFrame containing the 'preprocessed_content' column.
        n_features (int): The number of features to generate. Default is 10000.

    Returns:
        - dtm (DataFrame): Document-Term Matrix.
    """
    hashing_vectorizer = HashingVectorizer(n_features=n_features)
    dtm_matrix = hashing_vectorizer.fit_transform(df["preprocessed_content"])
    dtm = pd.DataFrame(
        dtm_matrix.toarray(),
        # columns=hashing_vectorizer.get_feature_names_out(),
        index=df.index,
    )
    return dtm
