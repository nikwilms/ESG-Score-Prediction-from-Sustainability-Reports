# SustainSight: NLP-Driven ESG Score Prediction
Welcome to SustainSight, your premier NLP-based tool for assessing a company's commitment to Environmental, Social, and Governance (ESG) principles. Leveraging state-of-the-art models like BERT and our custom-trained algorithms, we provide insightful ESG score predictions based on your uploaded sustainability reports.

This repository is useful for researchers, investors, and sustainability professionals who are interested in developing or using machine learning models to predict ESG scores.

#### -- Project Status: FINISHED

## The Innovators
[Marius Bosch](https://www.linkedin.com/in/marius-bosch-435158126/), [Selchuk Hadzhaahmed](https://www.linkedin.com/in/selchuk-hadzhaahmed-804379100/),
[Nikita Wilms](https://www.linkedin.com/in/nikita-wilms/)

## Why SustainSight?
Our tool is tailored for researchers, investors, and sustainability professionals who need a reliable, machine learning-driven method to predict ESG scores.

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
* Google Colab

## Project Outcome
The final ensemble model predicts ESG scores with an impressive accuracy, deviating by an average range of only 8.5% from actual ESG ratings.

## Future Work: Enhancing Transparency
We aim to incorporate features that make ESG performance transparent and actionable, providing not just scores but also insights into areas for improvement or validation.

## Where the Data Comes From
* [Yahoo API for ESG scores](https://pypi.org/project/yesg/): This module provides access to ESG scores for a variety of companies.
* [www.responsibilityreports.com](www.responsibilityreports.com) for ESG company reports: This website provides access to ESG company reports.

## Key Questions Addressed
* What are the underlying factors of a company's ESG score?
* Can we pinpoint features common among high-scoring sustainability reports?
* How does NLP contribute to predictive accuracy?
* How can the model be enhanced for better interpretability?

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


