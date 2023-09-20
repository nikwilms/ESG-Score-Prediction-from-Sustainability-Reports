import bs4 as bs
import requests
import pickle
import datetime as dt
import pandas as pd
import json
import urllib.request
import yesg

def getting_ESG_scores():
    """
    This function gets the ESG scores for the S&P 500 companies and saves them in a csv file.
    Return: dataframe with the ESG scores per year per company
    """
    # Getting resources from Wikipedia
    resource = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    # Parsing the resources
    soup = bs.BeautifulSoup(resource.text, 'html.parser')
    # Finding the table with the tickers
    table = soup.find('table', {'class': 'wikitable sortable'})

    # Creating an empty list for the tickers
    tickers = []
    # Finding all the rows in the table
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
    # Removing the \n from the tickers
    tickers = [s.replace('\n', '') for s in tickers]


    # Getting the ESG scores for each ticker
    dataframes = []
    for ticker in tickers:
        try:
            df = pd.DataFrame(yesg.get_historic_esg(ticker))
            df['Company_Symbol'] = ticker
            dataframes.append(df)
        except:
            pass
    # Concatenating the dataframes
    df = pd.concat(dataframes)

    df['timestamp'] = df.index
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    # Removing the non-values
    df.dropna(inplace=True)

    # Setting dataframe index to timestamp
    df['timestamp'] = df.index

    # Resetting the index
    df.reset_index(drop=True, inplace=True)

    # Setting the timestamp as datetime format
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    # Adding the year column to dataframe for calculating the average ESG score per year
    df['year'] = df['timestamp'].dt.year

    # Grouping the dataframe by year and ticker
    cleaned_df = df.groupby(['year', 'Company_Symbol']).mean()

    # Removing the timestamp column
    cleaned_df.drop(columns=['timestamp'], inplace=True)

    # Creating a csv file with the results
    cleaned_df.to_csv('/../data/SP500_EGS_Score_avarage_per_year.csv')

    esg_score = pd.read_csv('/../data/SP500_EGS_Score_avarage_per_year.csv')

    return esg_score