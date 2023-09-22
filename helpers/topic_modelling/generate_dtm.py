from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

def create_dtm(df, max_df=0.97, min_df=0.03):
    """
    Generate a Document-Term Matrix (DTM) using the preprocessed content in the DataFrame.
    
    Args:
    - df (DataFrame): DataFrame containing the 'preprocessed_content' column.
    - max_df (float): Ignore terms that have a document frequency higher than max_df. Default is 0.95.
    - min_df (float): Ignore terms that have a document frequency lower than min_df. Default is 0.05.
    
    Returns:
    - dtm (DataFrame): Document-Term Matrix.
    """
    count_vectorizer = CountVectorizer(max_df=max_df, min_df=min_df)
    dtm_matrix = count_vectorizer.fit_transform(df['preprocessed_content'])
    dtm = pd.DataFrame(dtm_matrix.toarray(), columns=count_vectorizer.get_feature_names_out(), index=df.index)
    return dtm

