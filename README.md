# SustainSight: Revolutionizing ESG Score Prediction with NLP
Welcome to SustainSight, your go-to NLP-powered tool for assessing a company's adherence to Environmental, Social, and Governance (ESG) principles. With cutting-edge technologies like BERT and custom-trained models, we provide immediate and accurate ESG score predictions based on your uploaded sustainability reports.
This repository is useful for researchers, investors, and sustainability professionals who are interested in developing or using machine learning models to predict ESG scores.

#### -- Project Status: FINISHED

## The Innovators
[Marius Bosch](https://www.linkedin.com/in/marius-bosch-435158126/), [Selchuk Hadzhaahmed](https://www.linkedin.com/in/selchuk-hadzhaahmed-804379100/),
[Nikita Wilms](https://www.linkedin.com/in/nikita-wilms/)

## Elevating ESG Transparency
Our mission is to make ESG performance transparent and actionable. Upload a sustainability report and receive a detailed ESG score, pinpointing areas for improvement or validation.

### Tech Stack & Techniques
* Web Scraping via Selenium
* Text Preprocessing
* Feature Engineering
* NLP (Natural Language Processing)
* N-Gram Analysis
* BERT Transformation
* LDA & TF-IDF
* LSTM Networks
* Dimensionality Reduction
* Ensemble Learning
* Regression Models: XGBoost, LGBM, Random Forest, Gradient Boosting, Lasso, Ridge

## Why This Project Matters
Success means delivering a machine learning model that not only predicts ESG scores accurately but also offers interpretability.

## Where the Data Comes From
* [Yahoo API for ESG scores](https://pypi.org/project/yesg/): This module provides access to ESG scores for a variety of companies.
* [www.responsibilityreports.com](www.responsibilityreports.com) for ESG company reports: This website provides access to ESG company reports.

## Key Questions
* What dictates a company's ESG score?
* Can we identify hallmark features in high-ESG-score reports?
* How can NLP extract predictive insights from text?
* How do we make the model interpretable?

## Challenges We Overcame
Interpretability is crucial. Especially for ESG scores, it's vital that users understand why a particular score was predicted.

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

The `requirements.txt` file contains the libraries needed for deployment.


