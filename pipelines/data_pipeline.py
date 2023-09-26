from helpers.text_preprocessing.preprocess_text import preprocess_text
from helpers.text_preprocessing.add_spelling_correction import add_spelling_correction
from helpers.ngrams.detect_ngrams import detect_ngrams
from helpers.ngrams.add_ngrams import add_ngrams
from helpers.pdf_to_text.remove_unnecessary_context_from_PDF import process_pdfs_in_directory
import pandas as pd

def data_pipeline(input_path, output_path='../data/extracted_text_sustainability_reports.csv'):
    
    # Stage 0: 
    process_pdfs_in_directory(input_path, '../data/extracted_text_sustainability_reports.csv')

    # 0.1 - Read in the CSV and create a DataFrame
    df = pd.read_csv('../data/extracted_text_sustainability_reports.csv')

    # Stage 1: Text Preprocessing
    df = preprocess_text(df)  # Note that we're directly passing the DataFrame

    # Stage 2: Adding spelling correction
    df = add_spelling_correction(df, output_folder='../data/')  # Note that we're directly passing the DataFrame

    # Stage 3: Detect ngrams
    df, bigram_model, trigram_model = detect_ngrams(df) # Assuming detect_ngrams returns DataFrame as first element in a tuple

    # Stage 4: Adding ngrams
    df["preprocessed_content"] = df["preprocessed_content"].apply(lambda x: add_ngrams(x, bigram_model=bigram_model, trigram_model=trigram_model))

    # Stage 5: Save the final dataframe to CSV
    df.to_csv('../data/ready_to_model_data/ready_to_model_df.csv')

    return df