from spellchecker import SpellChecker
import pandas as pd
import os
from multiprocessing import Pool

def correct_text(text):
    spell = SpellChecker()
    misspelled = spell.unknown(text.split())
    corrected_text = [spell.correction(word) if word in misspelled else word for word in text.split()]
    return " ".join(corrected_text)

def add_spelling_correction(df, output_folder="../../data/", n_processes=4):
    """
    Perform spell checking on the 'preprocessed_content' column of the DataFrame.
    Args:
    - df (pandas DataFrame): DataFrame containing the 'preprocessed_content' column.
    Returns:
    - df: DataFrame with spell-checked 'preprocessed_content'.
    """
    
    # Use parallel processing for spell correction
    with Pool(n_processes) as p:
        df["preprocessed_content"] = p.map(correct_text, df["preprocessed_content"].tolist())

    # Save the spell-checked DataFrame to a folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df.to_csv(os.path.join(output_folder, "spell_checked_df.csv"), index=False)

    return df

