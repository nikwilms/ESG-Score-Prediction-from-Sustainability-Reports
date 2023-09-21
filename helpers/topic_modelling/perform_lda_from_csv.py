import pandas as pd
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

def perform_lda_from_csv(file_path, num_topics=100):
    # Read CSV into a DataFrame
    df = pd.read_csv(file_path)
    
    # Tokenize the 'preprocessed_content' column
    tokenized_data = df['preprocessed_content'].apply(lambda x: x.split())
    
    # Create a Gensim dictionary from the tokenized data
    dictionary = corpora.Dictionary(tokenized_data)
    
    # Create a corpus from the dictionary
    corpus = [dictionary.doc2bow(text) for text in tokenized_data]
    
    # Create the LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=30)
    
    # Calculate coherence score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=tokenized_data, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    
    # Visualize topics
    lda_display = gensimvis.prepare(lda_model, corpus, dictionary)
    
    return lda_model, corpus, dictionary, coherence_lda, lda_display

# Usage
'''lda_model, corpus, dictionary, coherence_lda, lda_display = perform_lda_from_csv('path_to_your_data.csv')
print(f"Coherence Score: {coherence_lda}")
pyLDAvis.display(lda_display)'''
