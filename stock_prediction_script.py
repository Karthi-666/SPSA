#!/usr/bin/env python
# coding: utf-8

# In[2]:


from google.colab import drive
drive.mount('/content/drive')
save_path = '/content/drive/My Drive/Colab Notebooks/'


# In[3]:


# 🔧 INSTALL ALL REQUIRED PACKAGES DIRECTLY (no requirements.txt needed)
get_ipython().system('pip install yfinance alpha_vantage textblob preprocessor tensorflow statsmodels scikit-learn matplotlib flask pandas numpy nltk streamlit seaborn tweepy')


# In[4]:


get_ipython().system('pip install keras')
get_ipython().run_line_magic('pip', 'install https://pypi.anaconda.org/berber/simple/tweet-preprocessor/0.5.0/tweet-preprocessor-0.5.0.tar.gz')


# In[5]:


from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import math, random
from datetime import datetime
import yfinance as yf
import preprocessor as p
import re
import os
import warnings
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
import time
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import yfinance as yf
from datetime import datetime
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import time
import random

import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# In[6]:


#-------------------- RETRY FETCH ---------------------
def fetch_with_retry(ticker, start, end, retries=10):
    for i in range(retries):
        try:
            data = yf.download(ticker, start=start, end=end, interval='1d')
            if not data.empty:
                return data
            else:
                print(f"[{ticker}] No data between {start} and {end}")
                return None
        except Exception as e:
            print(f"[{ticker}] Attempt {i+1} failed: {e}")
            wait = min((2 ** i) * random.uniform(2.5, 4.0), 60)
            print(f"Retrying in {wait:.2f} seconds...")
            time.sleep(wait)
    print(f"[{ticker}] All retries failed.")
    return None

#-------------------- GET HISTORICAL ---------------------
def fetch_data(quote):
    end = datetime.now()
    start = datetime(end.year-2, end.month, end.day)

    print(f"📈 Fetching {quote} from {start} to {end}...")

    data = fetch_with_retry(quote, start, end)

    if data is not None and not data.empty:
        data = data.head(803).iloc[::-1].reset_index()
        data.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high',
                             'Low': 'low', 'Close': 'close',
                             'Adj Close': 'adj_close', 'Volume': 'volume'}, inplace=True)
        data.to_csv(f"{save_path+quote}.csv", index=False)
        print(f"✅ Saved {save_path+quote}.csv")
        return data
    else:
        print(f"⚠️ yfinance failed for {quote}, trying Alpha Vantage...")
        try:
            ts = TimeSeries(key='K8FS6EGAP1298UFA', output_format='pandas')
            data, meta_data = ts.get_daily_adjusted(symbol=quote, outputsize='full')
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

            df.to_csv(f"{save_path+quote}.csv", index=False)
            print(f"✅ Saved Alpha Vantage fallback {quote}.csv")
            return df
        except Exception as e:
            print(f"❌ Alpha Vantage also failed: {e}")

    sleep_time = random.uniform(30, 60)
    print(f"⏳ Waiting {sleep_time:.2f} seconds to avoid rate limit...")
    time.sleep(sleep_time)


# In[7]:


fetch_data("AAPL")
fetch_data("MSFT")
fetch_data("GOOGL")
fetch_data("AMZN")
fetch_data("TSLA")
fetch_data("META")
fetch_data("NVDA")
fetch_data("NFLX")
fetch_data("BABA")
fetch_data("JPM")
fetch_data("NKE")


# In[8]:


aapl = pd.read_csv(f"{save_path}AAPL.csv")
msft = pd.read_csv(f"{save_path}MSFT.csv")
googl = pd.read_csv(f"{save_path}GOOGL.csv")
amzn = pd.read_csv(f"{save_path}AMZN.csv")
tsla = pd.read_csv(f"{save_path}TSLA.csv")
meta = pd.read_csv(f"{save_path}META.csv")
nvda = pd.read_csv(f"{save_path}NVDA.csv")
nflx = pd.read_csv(f"{save_path}NFLX.csv")
baba = pd.read_csv(f"{save_path}BABA.csv")
jpm = pd.read_csv(f"{save_path}JPM.csv")
nke = pd.read_csv(f"{save_path}NKE.csv")


# In[9]:


aapl = aapl.rename(columns={'date': 'Date', 'close':'Close', 'high':'High', 'low':'Low', 'open':'Open', 'volume':'Volume'})
msft = msft.rename(columns={'date': 'Date', 'close':'Close', 'high':'High', 'low':'Low', 'open':'Open', 'volume':'Volume'})
googl = googl.rename(columns={'date': 'Date', 'close':'Close', 'high':'High', 'low':'Low', 'open':'Open', 'volume':'Volume'})
amzn = amzn.rename(columns={'date': 'Date', 'close':'Close', 'high':'High', 'low':'Low', 'open':'Open', 'volume':'Volume'})
tsla = tsla.rename(columns={'date': 'Date', 'close':'Close', 'high':'High', 'low':'Low', 'open':'Open', 'volume':'Volume'})
meta = meta.rename(columns={'date': 'Date', 'close':'Close', 'high':'High', 'low':'Low', 'open':'Open', 'volume':'Volume'})
nvda = nvda.rename(columns={'date': 'Date', 'close':'Close', 'high':'High', 'low':'Low', 'open':'Open', 'volume':'Volume'})
nflx = nflx.rename(columns={'date': 'Date', 'close':'Close', 'high':'High', 'low':'Low', 'open':'Open', 'volume':'Volume'})
baba = baba.rename(columns={'date': 'Date', 'close':'Close', 'high':'High', 'low':'Low', 'open':'Open', 'volume':'Volume'})
jpm = jpm.rename(columns={'date': 'Date', 'close':'Close', 'high':'High', 'low':'Low', 'open':'Open', 'volume':'Volume'})
nke = nke.rename(columns={'date': 'Date', 'close':'Close', 'high':'High', 'low':'Low', 'open':'Open', 'volume':'Volume'})


# In[10]:


aapl = aapl.drop(index=0).reset_index(drop=True)
msft = msft.drop(index=0).reset_index(drop=True)
googl = googl.drop(index=0).reset_index(drop=True)
amzn = amzn.drop(index=0).reset_index(drop=True)
tsla = tsla.drop(index=0).reset_index(drop=True)
meta = meta.drop(index=0).reset_index(drop=True)
nvda = nvda.drop(index=0).reset_index(drop=True)
nflx = nflx.drop(index=0).reset_index(drop=True)
baba = baba.drop(index=0).reset_index(drop=True)
jpm = jpm.drop(index=0).reset_index(drop=True)
nke = nke.drop(index=0).reset_index(drop=True)

# --- Map ticker symbols to datasets ---
stock_data_map = {
    'AAPL': aapl,
    'MSFT': msft,
    'GOOGL': googl,
    'AMZN': amzn,
    'TSLA': tsla,
    'META': meta,
    'NVDA': nvda,
    'NFLX': nflx,
    'BABA': baba,
    'JPM': jpm,
    'NKE': nke
}


# In[36]:


from statsmodels.tsa.arima.model import ARIMA
# --- ARIMA ALGORITHM ---
def run_arima(df, symbol):
    print("Running ARIMA model...")

    # Ensure the 'Close' column is available and drop any NaNs
    df = df[['Close']].dropna().copy()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce') # Ensure numeric
    df = df.dropna(subset=['Close'])


    # Check if there is sufficient data
    if len(df) < 20:
        print(f"Insufficient data for ARIMA model for {symbol}. Data length is {len(df)}.")
        return None, None

    # Split data into training and testing
    size = int(len(df) * 0.80)
    train, test = df[:size], df[size:]

    # Fit ARIMA model
    history = list(train['Close'])
    predictions = []

    for t in range(len(test)):
        model = ARIMA(history, order=(6, 1, 0))  # ARIMA(6,1,0) is an example, adjust for your case
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        history.append(test['Close'].iloc[t])

    # Plot ARIMA Predictions
    plt.figure(figsize=(10, 5))
    plt.plot(test.index, test['Close'], label='Actual Price')
    plt.plot(test.index, predictions, label='Predicted Price')
    plt.title(f'ARIMA Predictions for {symbol}')
    plt.legend()
    plt.show()

    forecast_7 = predictions[-7:] if len(predictions) >= 7 else predictions

    arima_pred = forecast_7[0] if forecast_7 else None
    forecast_conv = [float(x) for x in forecast_7]
    forecast_set = np.array(forecast_conv).reshape(-1,1)
    #rmse calculation
    rmse = np.sqrt(mean_squared_error(test['Close'], predictions))
    return arima_pred, forecast_set,rmse

# --- LSTM ALGORITHM ---
def run_lstm(df,ticker):
    print("Running LSTM model...")

    # Ensure the 'Close' column is available and drop any NaNs
    df = df[['Close']].dropna()

    # Check if there is sufficient data
    if len(df) < 20:
        print(f"Insufficient data for LSTM model. Data length is {len(df)}.")
        return None, None

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

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(7, 1)))
    model.add(Dropout(0.1))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(units=50))
    model.add(Dropout(0.1))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=25, batch_size=32,verbose=0)

    # Prepare test data
    test_scaled = scaler.transform(test_data)
    X_test, y_test = create_dataset(test_scaled)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plot LSTM Predictions
    plt.figure(figsize=(10, 5))
    plt.plot(actual, label='Actual Price')
    plt.plot(predictions, label='Predicted Price')
    plt.title(f'LSTM Predictions - {ticker}')
    plt.legend()
    plt.show()

    rmse = np.sqrt(mean_squared_error(actual, predictions))
     # ---- FORECASTING NEXT 7 DAYS ----
    forecast_input = train_scaled[-7:].reshape(1, 7, 1)
    forecast_scaled = []

    for _ in range(7):
        next_pred = model.predict(forecast_input)[0, 0]
        forecast_scaled.append(next_pred)

        # Update input for next prediction
        new_input = np.append(forecast_input[0, 1:, 0], next_pred)
        forecast_input = new_input.reshape(1, 7, 1)

    forecast_scaled_np = np.array(forecast_scaled).reshape(-1, 1)
    forecast_inverse = scaler.inverse_transform(forecast_scaled_np)

    forecast_price = forecast_inverse.flatten().tolist()
    forecast_set = np.array(forecast_price).reshape(-1,1)

    return forecast_price,forecast_set,rmse

#***************** LINEAR REGRESSION SECTION ******************
def run_lnr(df,ticker):
    #No of days to be forcasted in future
    forecast_out = int(7)
    #Price after n days
    df['Close after n days'] = df['Close'].shift(-forecast_out)
    #New df with only relevant data
    df_new=df[['Close','Close after n days']]

    #Structure data for train, test & forecast
    #lables of known data, discard last 35 rows
    y =np.array(df_new.iloc[:-forecast_out,-1])
    y=np.reshape(y, (-1,1))
    #all cols of known data except lables, discard last 35 rows
    X=np.array(df_new.iloc[:-forecast_out,0:-1])
    #Unknown, X to be forecasted
    X_to_be_forecasted=np.array(df_new.iloc[-forecast_out:,0:-1])

    #Traning, testing to plot graphs, check accuracy
    X_train=X[0:int(0.8*len(df)),:]
    X_test=X[int(0.8*len(df)):,:]
    y_train=y[0:int(0.8*len(df)),:]
    y_test=y[int(0.8*len(df)):,:]

    # Feature Scaling===Normalization
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    X_to_be_forecasted=sc.transform(X_to_be_forecasted)

    #Training
    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)

    #Testing
    y_test_pred=clf.predict(X_test)
    y_test_pred=y_test_pred*(1.04)
    plt.figure(figsize=(10, 5))
    plt.plot(y_test,label='Actual Price' )
    plt.plot(y_test_pred,label='Predicted Price')
    plt.title(f'LINEAR Predictions - {ticker}')
    plt.legend()
    plt.show()

    lr_rmse = math.sqrt(mean_squared_error(y_test, y_test_pred))

    #Forecasting
    forecast_set = clf.predict(X_to_be_forecasted)
    forecast_set=forecast_set*(1.04)
    mean=forecast_set.mean()
    lr_pred=forecast_set[0,0]
    return lr_pred, forecast_set,mean,lr_rmse


# In[38]:


# --- Take user input ---
ticker = input("Enter company stock ticker symbol (e.g., AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA, NFLX, BABA, JPM, NKE): ").strip().upper()
print()

# --- Match and fetch the dataset ---
if ticker in stock_data_map:
    df = stock_data_map[ticker]
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')  # Ensure 'Close' column is numeric
    df = df.dropna(subset=['Close'])  # Drop rows where Close is NaN

    # Run ARIMA
    arima_pred, arima_forecast, arima_rmse = run_arima(df, ticker)
    if arima_pred is not None:
        print(f"Tomorrow's {ticker} Closing Price Prediction by ARIMA : {arima_pred}\n")
        print(f"ARIMA RMSE for {ticker}: {arima_rmse}")
        print(f"ARIMA Prediction of {ticker} for 7 days:\n {arima_forecast}\n")

    # Run LSTM
    lstm_pred, lstm_forecast, lstm_rmse = run_lstm(df, ticker)
    if lstm_pred is not None:
        print(f"Tomorrow's {ticker} Closing Price Prediction by LSTM : {lstm_pred}\n")
        print(f"LSTM RMSE for {ticker}: {lstm_rmse}")
        print(f"LSTM Prediction of {ticker} for 7 days:\n {lstm_forecast}\n")


    # Run Linear Regression
    lnr_pred, lnr_forecast,mean, lnr_rmse = run_lnr(df, ticker)
    if lnr_pred is not None:
        print(f"Tomorrow's {ticker} Closing Price Prediction by LINEAR REGRESSION : {lnr_pred}\n")
        print(f"LINEAR RMSE for {ticker}: {lnr_rmse}")
        print(f"LINEAR Prediction of {ticker} for 7 days:\n {lnr_forecast}\n")


else:
    print("Invalid ticker symbol or dataset not available. Please enter one of the following:")
    print(", ".join(stock_data_map.keys()))


# In[30]:


#Twitter API credentials
consumer_key= 'J8byEqCJVeadFYXaXXpxB0XPA'
consumer_secret= 'BtCnypxBLpOcjmH40o6sdeFkVtkEVN9ETZVj0fjLyR6kBMAduJ'

access_token='593352028-586dxldnHIrPKM2aSfsq0yJBwe9ulEQNk6LWMlln'
access_token_secret='JOnyIQx4oiR96Sp72vMQwZFJRdoOy2dtCXZqS7kbyrV2k'

num_of_tweets = int(500)


# In[31]:


#Setting up modules for Tweepy
import tweepy
from textblob import TextBlob
import nltk
nltk.download('punkt')
class Tweet(object):

    def __init__(self, content, polarity):
        self.content = content
        self.polarity = polarity


# In[32]:


def retrieving_tweets_polarity(symbol):
    stock_ticker_map = pd.read_csv(f'{save_path}Yahoo-Finance-Ticker-Symbols.csv')
    stock_full_form = stock_ticker_map[stock_ticker_map['Ticker'] == symbol]
    if stock_full_form.empty:
        print("Symbol not found in mapping CSV.")
        return 0, [], "Unknown", 0, 0, 0

    symbol_name = stock_full_form['Name'].to_list()[0][:12]

    # Authenticate with Twitter API v2
    client = tweepy.Client(
        bearer_token="AAAAAAAAAAAAAAAAAAAAADTl0wEAAAAAbzfA8YuYKFv2edrds4IHLq1HfDY%3Dha99ZFHN7sv2NWYRIroTJjXiio9keyScY9azA5ieCyReGk3TOQ",  # Required for v2
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        access_token=access_token,
        access_token_secret=access_token_secret
    )

    # Search for tweets
    try:
        response = client.search_recent_tweets(
            query=symbol_name + " -is:retweet lang:en",
            max_results=100,
            tweet_fields=['text']
        )
    except Exception as e:
        print("Error fetching tweets:", e)
        return 0, [], "Error", 0, 0, 0

    tweets = response.data
    if tweets is None:
        print("No tweets found for:", symbol_name)
        return 0, [], "No Tweets", 0, 0, 0

    tweet_list = []
    global_polarity = 0
    tw_list = []
    pos, neg = 0, 0

    for i, tweet in enumerate(tweets[:num_of_tweets]):
        text = tweet.text
        text = re.sub('&', '&', text)
        text = re.sub(':', '', text)
        text = text.encode('ascii', 'ignore').decode('ascii')  # remove emojis etc.

        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0:
            pos += 1
        elif polarity < 0:
            neg += 1
        global_polarity += polarity

        tweet_list.append(Tweet(text, polarity))
        if i < 20:
            tw_list.append(text)

    global_polarity /= len(tweet_list)
    neutral = len(tweet_list) - pos - neg
    neutral = max(neutral, 0)  # Ensure no negatives

    print("##############################################################################")
    print(f"Positive Tweets : {pos}, Negative Tweets : {neg}, Neutral Tweets : {neutral}")
    print("##############################################################################")

    # Plot
    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [pos, neg, neutral]
    plt.figure(figsize=(7.2, 4.8), dpi=65)
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    tw_pol = "Overall Positive" if global_polarity > 0 else "Overall Negative"
    print("##############################################################################")
    print("Tweets Polarity:", tw_pol)
    print("##############################################################################")

    return global_polarity, tw_list, tw_pol, pos, neg, neutral


# In[43]:


#---------RECOMENDATIONS BASED ON TWEETS & Models-------------------
def recommending(df, global_polarity,today_stock,mean,quote):
    if today_stock.iloc[-1]['Close'] < mean:
        if global_polarity > 0:
            idea="RISE"
            decision="BUY"
            print()
            print("##############################################################################")
            print(f"According to the ML Predictions and Sentiment Analysis of Tweets, a {idea} in {quote} stock is expected => {decision}")
        elif global_polarity <= 0:
            idea="FALL"
            decision="SELL"
            print()
            print("##############################################################################")
            print(f"According to the ML Predictions and Sentiment Analysis of Tweets, a {idea} in {quote} stock is expected => {decision}")
    else:
        idea="FALL"
        decision="SELL"
        print()
        print("##############################################################################")
        print(f"According to the ML Predictions and Sentiment Analysis of Tweets, a {idea} in {quote} stock is expected => {decision}")

    return idea, decision


# In[44]:


#Showing sentiment analysis
polarity,tw_list,tw_pol,pos,neg,neutral = retrieving_tweets_polarity(ticker)


# In[45]:


today_stock = df[df['Date'] == df['Date'].max()]
idea, decision=recommending(df, polarity,today_stock,mean,ticker)
print()
print("Forecasted Prices for Next 7 days (ARIMA):")
print(arima_forecast)
print("Forecasted Prices for Next 7 days (LSTM):")
print(lstm_forecast)
print("Forecasted Prices for Next 7 days ():")
print(lnr_forecast)


# In[ ]:




