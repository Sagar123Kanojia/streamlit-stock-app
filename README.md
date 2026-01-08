 Stock Price Prediction Web App using Machine Learning

A Streamlit-based web application that predicts stock prices using Machine Learning models, supporting both Indian and US stock markets.
The app allows users to visualize historical stock data, analyze trends, and generate future price predictions using past market behavior.

 Project Overview

This project demonstrates an end-to-end Machine Learning workflow applied to the stock market domain:

 Data collection from real financial sources

 Data preprocessing and feature engineering

 ML model training and prediction

 Interactive visualization using Streamlit

It is built primarily for learning, analysis, and portfolio demonstration purposes.


 Key Features

  Supports both Indian & US stocks

  User-defined date range selection

  Automatic historical data fetching

  Machine Learning–based price prediction

  Interactive charts:

   Historical prices

   Predicted vs Actual prices

  Real-time stock-related news integration

  Clean and user-friendly Streamlit UI
  

 Tech Stack
 Programming & Frameworks
  Python
  Streamlit

Data & ML Libraries
 pandas
 numpy
 scikit-learn
 
Data Sources
 nsepy → Indian stock market data
 yfinance → US stock market data
 

How It Works (Workflow)

User Input

Select stock symbol (Indian or US)

Choose date range

Data Collection

Indian stocks → nsepy

US stocks → yfinance

Data Preprocessing

Handling missing values

Feature selection

Date-based indexing

Model Training

Regression-based ML model

Trained on historical price data

Prediction

Future stock price prediction

Comparison with actual prices

Visualization

Line charts for historical trends

Prediction vs Actual price plots

Display of recent financial news

 Machine Learning Model

Problem Type: Regression

Input Features:

Open price

High price

Low price

Close price

Volume

Output:

Predicted future stock price

⚠️ Note:
Stock price prediction is inherently uncertain.
This model is built for educational and analytical purposes only, not financial advice.

 How to Run the Project Locally
1️ Clone the Repository
2️ Install Dependencies (requirements.txt)
3️ Run the Streamlit App

 Author

Sagar Kanojia

Data Analytics & Machine Learning Enthusiast

Final Year Computer Science Student

 If you like this project, feel free to star ⭐ the repository!



