import pandas as pd
import os

def merge_dataframes(df1, df2, output_dir="../../data/model_data"):
    """
    Merges two dataframes on the 'ticker' and 'year' columns. 
    Returns the merged dataframe and lists of unmatched rows from both dataframes.
    Also saves the resulting dataframes as CSV files in the specified directory.
    """
    
    # Merge the dataframes on 'ticker' and 'year'
    model_data = pd.merge(df1, df2, left_on=['ticker', 'year'], right_on=['company_symbol', 'year'], how='inner')
    
    # Find unmatched rows
    unmatched_report_data = df1[~df1.ticker.isin(model_data.ticker)]
    unmatched_ESG_scores = df2[~df2.company_symbol.isin(model_data.company_symbol)]
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the dataframes as CSV
    model_data.to_csv(os.path.join(output_dir, "model_data.csv"), index=False)
    unmatched_report_data.to_csv(os.path.join(output_dir, "unmatched_report_data.csv"), index=True)
    unmatched_ESG_scores.to_csv(os.path.join(output_dir, "unmatched_ESG_scores.csv"), index=True)

    return model_data, unmatched_report_data, unmatched_ESG_scores

