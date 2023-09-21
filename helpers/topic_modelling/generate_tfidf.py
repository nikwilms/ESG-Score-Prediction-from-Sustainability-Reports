from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def create_tfidf(df, max_df=0.95, min_df=0.05):
    """
    Generate a TF-IDF matrix using the preprocessed content in the DataFrame.
    
    Args:
    - df (DataFrame): DataFrame containing the 'preprocessed_content' column.
    - max_df (float): Ignore terms that have a document frequency higher than max_df. Default is 0.95.
    - min_df (float): Ignore terms that have a document frequency lower than min_df. Default is 0.05.
    
    Returns:
    - tfidf (DataFrame): TF-IDF matrix.
    """
    tfidf_vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['preprocessed_content'])
    tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out(), index=df.index)
    return tfidf