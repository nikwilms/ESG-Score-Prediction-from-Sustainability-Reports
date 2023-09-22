from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def perform_nmf(df, n_topics=5, n_top_words=10):
    """
    Perform NMF-based topic modeling.
    
    Args:
        df (DataFrame): DataFrame containing the DTM or TF-IDF representation.
        n_topics (int): Number of topics to discover. Default is 5.
        n_top_words (int): Number of top words to display for each topic. Default is 10.
        
    Returns:
        W (ndarray): Document-topic matrix.
        H (ndarray): Topic-term matrix.
        feature_names (list): List of feature names (words/terms).
    """
    
    # Perform NMF
    nmf = NMF(n_components=n_topics, random_state=1).fit(df)
    
    # Document-topic matrix
    W = nmf.transform(df)
    
    # Topic-term matrix
    H = nmf.components_
    
    # Feature names
    feature_names = df.columns
    
    # Display topics
    print(f"Topics discovered via NMF (Top {n_top_words} words)")
    for topic_idx, topic in enumerate(H):
        print(f"Topic #{topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
    
    return W, H, feature_names

# Example usage:
# W, H, feature_names = perform_nmf(tfidf_matrix, n_topics=5, n_top_words=10)
