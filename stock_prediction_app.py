import streamlit as st
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import random
import math
import re
import os
import warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import nltk
from pyngrok import ngrok
import subprocess
nltk.download('punkt')

warnings.filterwarnings("ignore")
plt.style.use('ggplot')
ngrok.set_auth_token("2py0g4tbKYEiPihqIoDRp4xS0vd_77YcWS9bYjbeZGB6RVkwV")
# Set your save path here 
#from google.colab import drive
#drive.mount('/content/drive')
save_path = r'C:\Users\karth\Downloads\Stock-Prediction & Sentiment analysis project\UI'  # Adjust folder path for CSV storage

# Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = 'K8FS6EGAP1298UFA'

# ------------- Helper functions -------------

def fetch_with_retry(ticker, start, end, retries=5):
    for i in range(retries):
        try:
            data = yf.download(ticker, start=start, end=end, interval='1d')
            if not data.empty:
                return data
            else:
                st.warning(f"[{ticker}] No data between {start} and {end}")
                return None
        except Exception as e:
            wait = min((2 ** i) * random.uniform(2.5, 4.0), 60)
            st.warning(f"[{ticker}] Attempt {i+1} failed: {e}. Retrying in {wait:.2f}s")
            time.sleep(wait)
    st.error(f"[{ticker}] All retries failed.")
    return None

def fetch_data(ticker):
    end = datetime.now()
    start = datetime(end.year - 2, end.month, end.day)
    st.info(f"Fetching data for {ticker} from {start.date()} to {end.date()}...")

    data = fetch_with_retry(ticker, start, end)
    if data is not None and not data.empty:
        data = data.head(803).iloc[::-1].reset_index()
        data.rename(columns={'date': 'Date',
        'close': 'Close',
        'high': 'High',
        'low': 'Low',
        'open': 'Open',
        'volume': 'Volume'}, inplace=True)
        csv_path = f"{save_path}{ticker}.csv"
        data.to_csv(csv_path, index=False)
        st.success(f"Data saved to {csv_path}")
        return data
    else:
        st.warning(f"yfinance failed for {ticker}, trying Alpha Vantage...")
        try:
            ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
            data, _ = ts.get_daily_adjusted(symbol=ticker, outputsize='full')
            data = data.head(503).iloc[::-1].reset_index()
            df = pd.DataFrame({
                'date': data['date'],
                'open': data['1. open'],
                'high': data['2. high'],
                'low': data['3. low'],
                'close': data['4. close'],
                'adj_close': data['5. adjusted close'],
                'volume': data['6. volume']
            })
            csv_path = f"{save_path}{ticker}.csv"
            df.to_csv(csv_path, index=False)
            st.success(f"Alpha Vantage fallback data saved to {csv_path}")
            return df
        except Exception as e:
            st.error(f"Alpha Vantage also failed: {e}")
            return None

def load_data(ticker):
    try:
        df = pd.read_csv(f"{save_path}{ticker}.csv")
        st.success(f"Loaded saved data for {ticker}")
        return df
    except FileNotFoundError:
        st.warning(f"No saved data found for {ticker}, fetching now...")
        return fetch_data(ticker)

def preprocess_data(df):
    # Rename and clean columns for consistency
    df = df.rename(columns={
        'date': 'Date',
        'close': 'Close',
        'high': 'High',
        'low': 'Low',
        'open': 'Open',
        'volume': 'Volume'
    })
    df = df.drop(index=0, errors='ignore').reset_index(drop=True)
    return df

def plot_series(title, actual, predicted, xlabel='Time', ylabel='Price'):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(actual, label='Actual Price')
    ax.plot(predicted, label='Predicted Price')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    st.pyplot(fig)


# -------------- ARIMA model -----------------

def run_arima(df, ticker):
    st.info(f"Running ARIMA model for {ticker}...")

    df = df[['Close']].dropna().copy()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])

    if len(df) < 20:
        st.warning(f"Insufficient data for ARIMA model. Data length: {len(df)}")
        return None, None, None

    size = int(len(df) * 0.80)
    train, test = df[:size], df[size:]
    history = list(train['Close'])
    predictions = []

    for t in range(len(test)):
        model = ARIMA(history, order=(6,1,0))
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test['Close'].iloc[t])

    plot_series(f'ARIMA Predictions for {ticker}', test['Close'], predictions)

    forecast_7 = predictions[-7:] if len(predictions) >= 7 else predictions
    arima_pred = forecast_7[0] if forecast_7 else None
    forecast_set = np.array(forecast_7).reshape(-1,1)
    rmse = np.sqrt(mean_squared_error(test['Close'], predictions))

    st.success(f"ARIMA RMSE: {rmse:.4f}")
    return arima_pred, forecast_set, rmse

# -------------- LSTM model ------------------

def run_lstm(df, ticker):
    st.info(f"Running LSTM model for {ticker}...")

    df = df[['Close']].dropna().copy()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close']) 
    if len(df) < 20:
        st.warning(f"Insufficient data for LSTM model. Data length: {len(df)}")
        return None, None, None

    dataset = df.values
    training_size = int(len(dataset) * 0.80)
    train_data = dataset[:training_size]
    test_data = dataset[training_size - 7:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data)

    def create_dataset(data, time_step=7):
        X, y = [], []
        for i in range(time_step, len(data)):
            X.append(data[i-time_step:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_dataset(train_scaled)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(7,1)),
        Dropout(0.1),
        LSTM(units=50, return_sequences=True),
        Dropout(0.1),
        LSTM(units=50, return_sequences=True),
        Dropout(0.1),
        LSTM(units=50),
        Dropout(0.1),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0)

    test_scaled = scaler.transform(test_data)
    X_test, y_test = create_dataset(test_scaled)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y_test.reshape(-1,1))

    plot_series(f'LSTM Predictions for {ticker}', actual, predictions)

    rmse = np.sqrt(mean_squared_error(actual, predictions))
    st.success(f"LSTM RMSE: {rmse:.4f}")

    # Forecast next 7 days
    forecast_input = train_scaled[-7:].reshape(1,7,1)
    forecast_scaled = []
    for _ in range(7):
        next_pred = model.predict(forecast_input)[0,0]
        forecast_scaled.append(next_pred)
        new_input = np.append(forecast_input[0,1:,0], next_pred)
        forecast_input = new_input.reshape(1,7,1)

    forecast_set = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1,1))
    lstm_pred = forecast_set[0,0]

    return lstm_pred, forecast_set, rmse

#***************** LINEAR REGRESSION SECTION ******************
def run_lnr(df,ticker):
    forecast_out = 7
    df = df[['Close']].dropna().copy()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close']) 
    df['Close after n days'] = df['Close'].shift(-forecast_out)
    df_new = df[['Close','Close after n days']]

    y = np.array(df_new.iloc[:-forecast_out,-1])
    y = np.reshape(y, (-1,1))
    X = np.array(df_new.iloc[:-forecast_out,0:-1])
    X_to_be_forecasted = np.array(df_new.iloc[-forecast_out:,0:-1])

    X_train = X[0:int(0.8*len(df)),:]
    X_test = X[int(0.8*len(df)):,:]
    y_train = y[0:int(0.8*len(df)),:]
    y_test = y[int(0.8*len(df)):,:]

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_to_be_forecasted = sc.transform(X_to_be_forecasted)

    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)

    y_test_pred = clf.predict(X_test)
    y_test_pred = y_test_pred * 1.04
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label='Actual Price')
    plt.plot(y_test_pred, label='Predicted Price')
    plt.title(f'LINEAR Predictions - {ticker}')
    plt.legend()
    st.pyplot(plt)
    plt.clf()

    lr_rmse = math.sqrt(mean_squared_error(y_test, y_test_pred))

    forecast_set = clf.predict(X_to_be_forecasted)
    forecast_set = forecast_set * 1.04
    mean = forecast_set.mean()
    lr_pred = forecast_set[0,0]
    return lr_pred, forecast_set, mean, lr_rmse



class News(object):
    def __init__(self, content, polarity):
        self.content = content
        self.polarity = polarity


def retrieving_tweets_polarity(symbol):
    url = f"https://finviz.com/quote.ashx?t={symbol}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        req = requests.get(url, headers=headers)
        soup = BeautifulSoup(req.content, 'html.parser')

        news_table = soup.find('table', class_='fullview-news-outer')
        if not news_table:
            st.warning(f"No news found for symbol: {symbol}")
            return 0, [], "No News", 0, 0, 0
    
        rows = news_table.findAll('tr')
        news_list = []
        global_polarity = 0
        tw_list = []
        pos, neg = 0, 0

        for i, row in enumerate(rows[:20]):  # Limit to top 20 headlines
            text = row.a.get_text()
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            news_list.append(News(text, polarity))

        df_news = pd.DataFrame([{"content": news.content, "polarity": news.polarity} for news in news_list])

        global_polarity = df_news['polarity'].mean()
        pos = (df_news['polarity'] > 0).sum()
        neg = (df_news['polarity'] < 0).sum()
        neutral = len(df_news) - pos - neg
        tw_list = df_news['content'].tolist()

        st.write(f"Positive Tweets : {pos}, Negative Tweets : {neg}, Neutral Tweets : {neutral}")
        labels = ['Positive', 'Negative', 'Neutral']
        sizes = [pos, neg, neutral]
        fig1, ax1 = plt.subplots(figsize=(7.2, 4.8), dpi=65)
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)

        news_pol = "Overall Positive" if global_polarity > 0 else "Overall Negative"
        st.write(f"News Polarity: {news_pol}")

        return global_polarity, tw_list, news_pol, pos, neg, neutral
    
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return 0, [], "Error", 0, 0, 0


def recommending(global_polarity, today_stock, mean, quote):
    if today_stock.iloc[-1]['Close'] < mean:
        if global_polarity > 0:
            idea = "RISE"
            decision = "BUY"
        else:
            idea = "FALL"
            decision = "SELL"
    else:
        idea = "FALL"
        decision = "SELL"

    st.info(f"According to ML predictions and sentiment analysis of tweets, a {idea} in {quote} stock is expected => {decision}")
    return idea, decision



# ----------- Streamlit UI -------------

st.title("Stock Price Prediction with ARIMA, LSTM & Linear Regression")

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN','TSLA','META','NVDA','NFLX','BABA','JPM','JKE']
ticker = st.selectbox("Select stock ticker:", tickers)

action = st.radio("Choose action:", ["Load Saved Data", "Fetch New Data"])

if st.button("Load / Fetch Data"):
    if action == "Load Saved Data":
        df = load_data(ticker)  # Your function to load saved data
    else:
        df = fetch_data(ticker)  # Your function to fetch data

    if df is not None and not df.empty:
        st.write(df.tail())

        # ARIMA
        if st.checkbox("Run ARIMA model"):
            arima_pred, arima_forecast_set, arima_rmse = run_arima(df, ticker)
            if arima_pred:
                st.write(f"ARIMA next day predicted price: {arima_pred:.2f}")
                st.write(f"ARIMA 7-day forecast:\n{arima_forecast_set.flatten()}")

        # LSTM
        if st.checkbox("Run LSTM model"):
            lstm_pred, lstm_forecast_set, lstm_rmse = run_lstm(df, ticker)
            if lstm_pred:
                st.write(f"LSTM next day predicted price: {lstm_pred:.2f}")
                st.write(f"LSTM 7-day forecast:\n{lstm_forecast_set.flatten()}")

        # Linear Regression
        if st.checkbox("Run Linear Regression model"):
            lr_pred, lr_forecast_set, mean, lr_rmse = run_lnr(df, ticker)
            if lr_pred:
                st.write(f"Linear Regression next day predicted price: {lr_pred:.2f}")
                st.write(f"Linear Regression 7-day forecast:\n{lr_forecast_set.flatten()}")
                st.write(f"Mean forecasted price: {mean:.2f}")
                st.write(f"Linear Regression RMSE: {lr_rmse:.4f}")

        # Sentiment Analysis & Recommendation
        
        if st.checkbox("Run Sentiment Analysis (FinViz.com) and Recommendation"):
            polarity, headlines, summary, pos, neg, neutral = retrieving_tweets_polarity(ticker)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # coerce invalid strings to NaT
            df = df.dropna(subset=['Date'])  # optional: drop rows where Date couldn't be parsed
            today_stock = df[df['Date'] == df['Date'].max()]
            idea, decision = recommending(polarity, today_stock, mean if 'mean' in locals() else df['Close'].mean(), ticker)

else:
    st.info("Please select ticker and click 'Load / Fetch Data'")


# Step 4: Expose Streamlit on ngrok
public_url = ngrok.connect(port=8501,proto="http")
print("Public URL:", public_url)


