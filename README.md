# NER and Feature Engineering Project

## Overview

This project performs **Named Entity Recognition (NER)** and **feature engineering** on news articles to predict article popularity. The goal is to analyze articles from various sources (such as Politifact and GossipCop), extract useful features like named entities, sentiment scores, and engagement metrics, and build a predictive model to forecast article popularity.

### Key Components:

1. **Preprocessing**: Cleaning and preparing the data.
2. **NER and Feature Engineering**: Extracting named entities (persons, organizations, locations) and creating features such as article length, sentiment scores, and engagement metrics.
3. **Predictive Modeling**: Training machine learning models such as Linear Regression or Random Forest to predict article popularity using engineered features.
4. **Visualization**: Creating visualizations that show the relationship between named entities and article popularity.
5. **Model Evaluation**: Evaluating model performance using metrics like **Mean Absolute Error (MAE)** and **R-squared (R²)**.

## Directory Structure

- `data/`: Contains the input datasets (`*.csv`) with news articles and NER results.
- `notebooks/`: Jupyter notebooks for analysis, model training, and documentation.
- `output/`: Stores processed data, visualizations, and model outputs.
- `scripts/`: Reusable scripts for preprocessing, NER, feature engineering, and predictive modeling.
- `output/visualizations`: Stores generated visualizations.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Irbaaz-Patel-13/NER_Feature_Engineering.git
   ```
