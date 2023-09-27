import bs4 as bs
import requests
import pickle
import datetime as dt
import pandas as pd
import json
import urllib.request
import yesg

"""
    Created by: @CukoF
    Created on: 26.09.2023
    This file contains the functions for getting and calculating the ESG scores for the S&P 500 companies.
"""


def getting_ESG_scores():
    """
    This function gets the ESG scores for the S&P 500 companies and saves them in a csv file.
    Return: dataframe with the ESG scores per year per company
    """
    # Getting resources from Wikipedia
    resource = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    # Parsing the resources
    soup = bs.BeautifulSoup(resource.text, "html.parser")
    # Finding the table with the tickers
    table = soup.find("table", {"class": "wikitable sortable"})

    # Creating an empty list for the tickers
    tickers = []
    # Finding all the rows in the table
    for row in table.findAll("tr")[1:]:
        ticker = row.findAll("td")[0].text
        tickers.append(ticker)
    # Removing the \n from the tickers
    tickers = [s.replace("\n", "") for s in tickers]

    # Getting the ESG scores for each ticker
    dataframes = []
    for ticker in tickers:
        try:
            df = pd.DataFrame(yesg.get_historic_esg(ticker))
            df["Company_Symbol"] = ticker
            dataframes.append(df)
        except:
            pass
    # Concatenating the dataframes
    df = pd.concat(dataframes)

    df["timestamp"] = df.index
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    # Removing the non-values
    df.dropna(inplace=True)

    # Setting dataframe index to timestamp
    df["timestamp"] = df.index

    # Resetting the index
    df.reset_index(drop=True, inplace=True)

    # Setting the timestamp as datetime format
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    # Adding the year column to dataframe for calculating the average ESG score per year
    df["year"] = df["timestamp"].dt.year

    return df


def copy_df(df):
    first_df = df.copy()
    return first_df


first_df = copy_df(getting_ESG_scores())


def calculate_old_scores(first_df, df):
    """
    This function calculates the ESG scores for the S&P 500 companies before 2019-11-01.
    ESG Scores calculated by: (old ESG score * new ESG score) / old ESG score per year
    Step 1:
    - Getting the ESG scores for the S&P 500 companies before 2019-11-01
    Step 2:
    - Dividing the dataframe into 6 dataframes per year
    Step 3:
    - Calculating the ESG scores for each year
    Step 4:
    - Merging and organizing the dataframes into one dataframe
    Step 5:
    - Getting a new dataframe with current scoring system

    Args:
        first_df (DataFrame): Copy of the dataframe with the ESG scores for the S&P 500 companies
        df (DataFrame): Getting the ESG scores for the S&P 500 companies
    Returns:
        DataFrame: Calculated and cleaned dataframe with the ESG scores per year per company
    """
    df_11_2019 = df.query('timestamp == "2019-11-01"').reset_index(drop=True)
    df_11_2019 = df_11_2019.add_suffix("_old")

    df_first_2020 = df.query('timestamp == "2020-01-01"').reset_index(drop=True)
    df_first_2020 = df_first_2020.add_suffix("_new")

    before_11_19_df = df.query('timestamp < "2019-11-01"').reset_index(drop=True)
    before_11_19_df = before_11_19_df.groupby(["year", "Company_Symbol"]).mean()
    before_11_19_df.reset_index(inplace=True)

    years = [2014, 2015, 2016, 2017, 2018, 2019]
    old_datasets = []
    for year in years:
        old_year_df = before_11_19_df.loc[before_11_19_df["year"] == year]
        old_datasets.append(old_year_df)

    df_2014 = pd.DataFrame(old_datasets[0])
    df_2014 = df_2014.add_suffix("_2014")

    merged_calculation = pd.merge(
        df_11_2019,
        df_first_2020,
        left_on="Company_Symbol_old",
        right_on="Company_Symbol_new",
        how="inner",
    )
    merged_calculation.drop(
        columns=[
            "Company_Symbol_new",
            "timestamp_new",
            "timestamp_old",
            "year_old",
            "year_new",
        ],
        inplace=True,
    )

    merged_2014 = pd.merge(
        merged_calculation,
        df_2014,
        left_on="Company_Symbol_old",
        right_on="Company_Symbol_2014",
        how="inner",
    )
    merged_2014["Total-Score_after"] = (
        (merged_2014["Total-Score_old"] * merged_2014["Total-Score_new"])
        / merged_2014["Total-Score_2014"]
    ).round(2)
    merged_2014["E-Score_after"] = (
        (merged_2014["E-Score_old"] * merged_2014["E-Score_new"])
        / merged_2014["E-Score_2014"]
    ).round(2)
    merged_2014["S-Score_after"] = (
        (merged_2014["S-Score_old"] * merged_2014["S-Score_new"])
        / merged_2014["S-Score_2014"]
    ).round(2)
    merged_2014["G-Score_after"] = (
        (merged_2014["G-Score_old"] * merged_2014["G-Score_new"])
        / merged_2014["G-Score_2014"]
    ).round(2)

    df_2014_new = merged_2014[
        [
            "Company_Symbol_old",
            "Total-Score_after",
            "E-Score_after",
            "S-Score_after",
            "G-Score_after",
            "year_2014",
        ]
    ]
    df_2014_new.rename(
        columns={"Company_Symbol_old": "Company_Symbol", "year_2014": "year"},
        inplace=True,
    )

    df_2015 = pd.DataFrame(old_datasets[1])
    df_2015 = df_2015.add_suffix("_2015")

    df_2016 = pd.DataFrame(old_datasets[2])
    df_2016 = df_2016.add_suffix("_2016")

    df_2017 = pd.DataFrame(old_datasets[3])
    df_2017 = df_2017.add_suffix("_2017")

    df_2018 = pd.DataFrame(old_datasets[4])
    df_2018 = df_2018.add_suffix("_2018")

    df_2019 = pd.DataFrame(old_datasets[5])
    df_2019 = df_2019.add_suffix("_2019")

    def new_esg_calculation(df, merged_calculation, year):
        merged_year = pd.merge(
            merged_calculation,
            df,
            left_on="Company_Symbol_old",
            right_on="Company_Symbol_" + str(year),
            how="inner",
        )

        merged_year["Total-Score_after"] = (
            (merged_year["Total-Score_old"] * merged_year["Total-Score_new"])
            / merged_year["Total-Score_" + str(year)]
        ).round(2)
        merged_year["E-Score_after"] = (
            (merged_year["E-Score_old"] * merged_year["E-Score_new"])
            / merged_year["E-Score_" + str(year)]
        ).round(2)
        merged_year["S-Score_after"] = (
            (merged_year["S-Score_old"] * merged_year["S-Score_new"])
            / merged_year["S-Score_" + str(year)]
        ).round(2)
        merged_year["G-Score_after"] = (
            (merged_year["G-Score_old"] * merged_year["G-Score_new"])
            / merged_year["G-Score_" + str(year)]
        ).round(2)

        df_new_year = merged_year[
            [
                "Company_Symbol_old",
                "Total-Score_after",
                "E-Score_after",
                "S-Score_after",
                "G-Score_after",
                "year_" + str(year),
            ]
        ]

        df_new_year.rename(
            columns={"Company_Symbol_old": "Company_Symbol", "year_2014": "year"},
            inplace=True,
        )

        return df_new_year

    df_list = [df_2015, df_2016, df_2017, df_2018, df_2019]
    df_year = [2015, 2016, 2017, 2018, 2019]
    list_of_df = []
    for df, year in zip(df_list, df_year):
        list_of_df.append(new_esg_calculation(df, merged_calculation, year))

    df_2015_new = list_of_df[0]
    df_2016_new = list_of_df[1]
    df_2017_new = list_of_df[2]
    df_2018_new = list_of_df[3]
    df_2019_new = list_of_df[4]

    df_2014_new.rename(
        columns={
            "Total-Score_after": "Total-Score",
            "E-Score_after": "E-Score",
            "G-Score_after": "G-Score",
            "S-Score_after": "S-Score",
        },
        inplace=True,
    )

    df_2015_new.rename(
        columns={
            "Total-Score_after": "Total-Score",
            "E-Score_after": "E-Score",
            "G-Score_after": "G-Score",
            "S-Score_after": "S-Score",
            "year_2015": "year",
        },
        inplace=True,
    )

    df_2016_new.rename(
        columns={
            "Total-Score_after": "Total-Score",
            "E-Score_after": "E-Score",
            "G-Score_after": "G-Score",
            "S-Score_after": "S-Score",
            "year_2016": "year",
        },
        inplace=True,
    )

    df_2017_new.rename(
        columns={
            "Total-Score_after": "Total-Score",
            "E-Score_after": "E-Score",
            "G-Score_after": "G-Score",
            "S-Score_after": "S-Score",
            "year_2017": "year",
        },
        inplace=True,
    )

    df_2018_new.rename(
        columns={
            "Total-Score_after": "Total-Score",
            "E-Score_after": "E-Score",
            "G-Score_after": "G-Score",
            "S-Score_after": "S-Score",
            "year_2018": "year",
        },
        inplace=True,
    )

    df_2019_new.rename(
        columns={
            "Total-Score_after": "Total-Score",
            "E-Score_after": "E-Score",
            "G-Score_after": "G-Score",
            "S-Score_after": "S-Score",
            "year_2019": "year",
        },
        inplace=True,
    )

    new_before_11_19_df = pd.concat(
        [df_2014_new, df_2015_new, df_2016_new, df_2017_new, df_2018_new, df_2019_new]
    )

    after_11_19_df = first_df.query('timestamp > "2019-12-01"').reset_index(drop=True)

    after_11_19_df = (
        after_11_19_df.groupby(["year", "Company_Symbol"]).mean().reset_index()
    )

    after_11_19_df.drop(columns=["timestamp"], inplace=True)

    after_11_19_df[["Total-Score", "E-Score", "S-Score", "G-Score"]] = after_11_19_df[
        ["Total-Score", "E-Score", "S-Score", "G-Score"]
    ].round(2)

    after_11_19_df = after_11_19_df[
        [col for col in after_11_19_df.columns if col != "year"] + ["year"]
    ]

    new_scores = pd.concat([new_before_11_19_df, after_11_19_df])

    new_scores = new_scores[
        ["year"] + [col for col in new_scores.columns if col != "year"]
    ]

    new_scores.to_csv("SP500_EGS_Score_avarage_per_year.csv")

    esg_score = pd.read_csv("SP500_EGS_Score_avarage_per_year.csv")

    return esg_score
