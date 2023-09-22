import gensim
from gensim import corpora
from gensim.models import CoherenceModel
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG)  # Set the logging level to DEBUG


def objective(trial):
    """
    An objective function for LDA topic modeling.

    Args:
        trial: A Trial object from Optuna.

    Returns:
        The coherence score of the LDA model, corpus and dictionary.
    """

    # Read CSV into a DataFrame
    df = pd.read_csv("../data/lda_test_df.csv")
    logging.info(f"Trial {trial.number} started")

    # Tokenize the 'preprocessed_content' column
    tokenized_data = df["preprocessed_content"].apply(lambda x: x.split())

    # Create a Gensim dictionary from the tokenized data
    dictionary = corpora.Dictionary(tokenized_data)

    # Filter extremes
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=None)

    # Recreate the corpus using the filtered dictionary
    corpus = [dictionary.doc2bow(text) for text in tokenized_data]

    '''    # Create a corpus from the dictionary
    corpus = [dictionary.doc2bow(text) for text in tokenized_data]

    # No_below: Tokens that appear in less than 5 documents are filtered out.
    # No_above: Tokens that appear in more than 50% of the total corpus are also removed as default.
    # Keep_n: We limit ourselves to the top 1000 most frequent tokens (default is 100.000). Set to ‘None’ if you want to keep all.
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=None)'''

    # Suggest hyperparameters
    alpha = trial.suggest_float("alpha", 0.01, 1)
    eta = trial.suggest_float("eta", 0.01, 1)
    ntopics = trial.suggest_int("num_topics", 10, 50)

    # Train the LDA model
    model = gensim.models.LdaMulticore(
        workers=7,
        corpus=corpus,
        id2word=dictionary,
        num_topics=ntopics,
        random_state=100,
        passes=5,
        alpha=alpha,
        eta=eta,
    )

    # Calculate coherence score
    coherence_model_lda = CoherenceModel(
        model=model, texts=tokenized_data, dictionary=dictionary, coherence="c_v"
    )
    coherence_lda = coherence_model_lda.get_coherence()

    # Return coherence score
    return coherence_lda, corpus, dictionary
