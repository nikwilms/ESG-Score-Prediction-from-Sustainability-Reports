from spellchecker import SpellChecker
import pandas as pd
import os


def spell_check_df(df, output_folder="../../data/"):
    """
    Perform spell checking on the 'preprocessed_content' column of the DataFrame.
    Args:
    - df (pandas DataFrame): DataFrame containing the 'preprocessed_content' column.
    Returns:
    - df: DataFrame with spell-checked 'preprocessed_content'.
    """
    # Initialize SpellChecker
    spell = SpellChecker()

    # Perform spell correction and update the DataFrame
    df["preprocessed_content"] = df["preprocessed_content"].apply(
        lambda text: " ".join(
            [spell.correction(word) for word in text.split() if word is not None]
        )
    )

    # Save the spell-checked DataFrame to a folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df.to_csv(os.path.join(output_folder, "spell_checked_df.csv"), index=False)

    return df