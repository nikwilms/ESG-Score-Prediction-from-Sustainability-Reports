from concurrent.futures import ThreadPoolExecutor
import yfinance as yf
import pandas as pd


def fetch_data_for_row(named_tuple_row):
    """
    Fetch financial data for a single row represented by a named tuple containing ticker and year.

    Parameters:
    - named_tuple_row (namedtuple): A named tuple containing 'ticker' and 'year'.

    Returns:
    - DataFrame: A DataFrame containing the fetched data, empty DataFrame if an error occurs.
    """

    ticker = "Unknown"  # Initialize with a default value
    year = "Unknown"  # Initialize with a default value

    try:
        ticker = named_tuple_row.ticker
        year = named_tuple_row.year

        yf_ticker = yf.Ticker(ticker)
        financials = yf_ticker.financials
        cashflow = yf_ticker.cashflow
        balance = yf_ticker.balance_sheet
        info = {k: v for k, v in yf_ticker.info.items() if isinstance(v, (int, float))}

        financials = financials.loc[
            :, pd.to_datetime(financials.columns).year == year
        ].transpose()
        cashflow = cashflow.loc[
            :, pd.to_datetime(cashflow.columns).year == year
        ].transpose()
        balance = balance.loc[
            :, pd.to_datetime(balance.columns).year == year
        ].transpose()
        info_df = pd.DataFrame([info])

        financials.columns = "financials_" + financials.columns.astype(str)
        cashflow.columns = "cashflow_" + cashflow.columns.astype(str)
        balance.columns = "balance_" + balance.columns.astype(str)
        info_df.columns = "info_" + info_df.columns.astype(str)

        merged_data = pd.concat([financials, cashflow, balance, info_df], axis=1)
        merged_data["ticker"] = ticker
        merged_data["year"] = year

        return merged_data.reset_index(drop=True)
    except Exception as e:
        print(
            f"An error occurred in fetch_data_for_row for ticker: {ticker} and year: {year}. Error: {e}"
        )
        return pd.DataFrame()


def fetch_and_merge_data(df):
    """
    Fetch financial data for a DataFrame containing multiple rows of tickers and years.
    This function uses multi-threading to speed up the data fetching process.

    Parameters:
    - df (DataFrame): A DataFrame containing 'ticker' and 'year' columns.

    Returns:
    - DataFrame: A DataFrame containing the fetched data merged with the original DataFrame.
    """
    try:
        # Initialize an empty list to store fetched data
        fetched_data_list = []

        with ThreadPoolExecutor() as executor:
            fetched_data_list = list(
                executor.map(fetch_data_for_row, df.itertuples(index=False))
            )

        # Concatenate all the fetched data
        new_data = pd.concat(
            [data.iloc[[0]] for data in fetched_data_list if not data.empty],
            ignore_index=True,
        )

        # Debug: Print the shape and columns of new_data
        print(f"new_data shape: {new_data.shape}, columns: {new_data.columns}")

        # Merge new_data with df based on 'ticker' and 'year'
        final_df = pd.merge(df, new_data, on=["ticker", "year"], how="left")

        return final_df
    except Exception as e:
        print(f"An error occurred in fetch_and_merge_data: {e}")
        return df  # Return the original DataFrame as a fallback


# Example usage
# df = pd.DataFrame({'ticker': ['AAPL', 'GOOGL'], 'year': [2020, 2021]})
# final_df = fetch_and_merge_data(df)
