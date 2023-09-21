from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def create_dtm(df):
    """
    Create a Document-Term Matrix (DTM).
    
    Args:
    - df (pandas DataFrame): DataFrame containing a 'preprocessed_content' column with text data.
    
    Returns:
    - dtm (sparse matrix): Document-Term Matrix.
    - feature_names (list): List of feature names (words).
    """
    vectorizer = CountVectorizer()
    dtm = vectorizer.fit_transform(df['preprocessed_content'])
    feature_names = vectorizer.get_feature_names_out()
    return dtm, feature_names

# Example usage
# df = pd.read_csv("your_dataframe.csv")
# dtm, feature_names = create_dtm(df)
