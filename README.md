# ESG Score Prediction using Sustainability Reports
This repository contains code and data to train and evaluate a machine learning model to predict ESG scores from sustainability reports. The model is trained on a dataset of sustainability reports and ESG scores from a variety of companies. It can be used to predict ESG scores for new companies or to update ESG scores for existing companies.

This repository is useful for researchers, investors, and sustainability professionals who are interested in developing or using machine learning models to predict ESG scores.

#### -- Project Status: In Progress

## Team
[Marius Bosch](https://www.linkedin.com/in/marius-bosch-435158126/), [Selchuk Hadzhaahmed](https://www.linkedin.com/in/selchuk-hadzhaahmed-804379100/),
[Nikita Wilms](https://www.linkedin.com/in/nikita-wilms/)

## Project Objective
Disclose their ESG performance in a transparent and comprehensive way. The ESG score prediction model can be used to ensure that the ESG report is accurate and complete. It can also be used to identify areas where the company needs to provide more information about its ESG performance.

### Methods Used
* Scraping the data
* Text preprocessing
* Feature engineering
* Data Visualization
* Exploratory Data Analysis
* Natural language processing (NLP)

## Project Overview
The project will be successful if it is able to develop a machine learning model that can accurately predict ESG scores from sustainability reports. The model should also be interpretable, so that users can understand how the model makes its predictions.

## Data Sources
* [Yahoo API for ESG scores](https://pypi.org/project/yesg/): This module provides access to ESG scores for a variety of companies.
* [www.responsibilityreports.com](www.responsibilityreports.com) for ESG company reports: This website provides access to ESG company reports.

## Questions and Hypotheses

## Blockers and Challenges
It is important to be able to interpret the machine learning model so that you can understand how it makes its predictions. This is especially important for ESG scores, as it is important to be able to explain to users why the model predicted a particular score.

## Requirements:

- pyenv with Python: 3.11.3

### Setup

Use the requirements file in this repo to create a new environment.

```BASH
make setup

#or

pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_dev.txt
```

The `requirements.txt` file contains the libraries needed for deployment.. of model or dashboard .. thus no jupyter or other libs used during development.


