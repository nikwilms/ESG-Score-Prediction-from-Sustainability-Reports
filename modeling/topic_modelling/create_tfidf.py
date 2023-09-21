from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf(df):
    """
    Create a TF-IDF representation of the text data.
    
    Args:
    - df (pandas DataFrame): DataFrame containing a 'preprocessed_content' column with text data.
    
    Returns:
    - tfidf_matrix (sparse matrix): TF-IDF representation.
    - feature_names (list): List of feature names (words).
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['preprocessed_content'])
    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names

# Example usage
# df = pd.read_csv("your_dataframe.csv")
# tfidf_matrix, feature_names = create_tfidf(df)