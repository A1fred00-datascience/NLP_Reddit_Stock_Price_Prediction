# Stock Market Prediction from Reddit News Headlines

## Overview
This project explores the impact of news headlines on stock market trends, specifically analyzing the relationship between the sentiment of top headlines from Reddit's WorldNews channel and the Dow Jones Industrial Average (DJIA) movements. The aim is to identify if news sentiment can predict stock price fluctuations.

## Dataset Description
- **RedditNews.csv**: Contains historical news headlines from Reddit's WorldNews channel, with top 25 headlines for each date from June 8, 2008, to July 1, 2016.
- **DJIA_table.csv**: Provides daily Dow Jones Industrial Average (DJIA) trading data for the same period.
- **Combined_News_DJIA.csv**: A combined dataset including the date, daily DJIA movement label (1 for up, 0 for down), and top 25 news headlines.

## Objectives
- Perform sentiment analysis on news headlines using Natural Language Processing (NLP) techniques.
- Investigate the correlation between news sentiment and DJIA stock movements.
- Develop a predictive model to forecast stock trends based on headline sentiment.

## Methodology
1. **Data Preprocessing**: Cleaning and merging datasets for analysis.
2. **Exploratory Data Analysis (EDA)**: Uncovering patterns and testing hypotheses through statistical analysis and visualization.
3. **Sentiment Analysis**: Analyzing headlines sentiment and its correlation with stock movements.
4. **Feature Engineering**: Creating features from the headlines to use in predictive modeling.
5. **Model Development**: Building machine learning models to predict stock market trends from sentiment scores.
6. **Insights and Recommendations**: Providing insights and actionable recommendations based on model findings.

## Tools and Technologies
- Python for data processing and model development.
- Pandas and NumPy for data manipulation.
- Matplotlib and Seaborn for visualization.
- Scikit-learn for machine learning.
- NLTK or TextBlob for sentiment analysis.

## Conclusion
This project is a deep dive into how news sentiment correlates with stock market behaviors. Through comprehensive analysis and machine learning, it offers insights into the predictive power of news sentiment on stock movements, presenting a novel approach to financial market prediction.

**Note:** This project is based on my personal data exploration and analysis journey. It is for educational purposes and not intended as financial advice.

---

*This project reflects my fascination with the intersection of finance, news media, and data science. The methodologies and findings herein are my contributions to understanding the complex dynamics at play in financial markets influenced by global news sentiment.*
