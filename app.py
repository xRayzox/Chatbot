import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, LSTM
from datetime import date, datetime, timedelta
from stocknews import StockNews



# --- SIDEBAR CODE
ticker = st.sidebar.selectbox('Select your Crypto', ["BTC-USD", "ETH-USD"])

start_date = st.sidebar.date_input('Start Date', date.today() - timedelta(days=365))
end_date = st.sidebar.date_input('End Date')


# --- MAIN PAGE
st.header('Cryptocurrency Prediction')

col1, col2, = st.columns([1,9])
with col1:
  st.image('icons/'+ ticker +'.png', width=75)
with col2:
  st.write(f" ## { ticker}")

ticker_obj = yf.Ticker(ticker)


# --- CODE

model_data = ticker_obj.history(interval='1h', start=start_date, end=end_date)

# Extract the 'close' column for prediction
target_data = model_data["Close"].values.reshape(-1, 1)

# Normalize the target data
scaler = MinMaxScaler()
target_data_normalized = scaler.fit_transform(target_data)

# Normalize the input features
input_features = ['Open', 'High', 'Low', 'Volume']
input_data = model_data[input_features].values
input_data_normalized = scaler.fit_transform(input_data)

def build_lstm_model(input_data, output_size, neurons, activ_func='linear', dropout=0.2, loss='mse', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)

    return model


# Hyperparameters
np.random.seed(245)
window_len = 10
split_ratio = 0.8  # Ratio of training set to total data
zero_base = True
lstm_neurons = 50
epochs = 100
batch_size = 128 #32
loss = 'mean_squared_error'
dropout = 0.24
optimizer = 'adam'

def extract_window_data(input_data, target_data, window_len):
    X = []
    y = []
    for i in range(len(input_data) - window_len):
        X.append(input_data[i : i + window_len])
        y.append(target_data[i + window_len])
    return np.array(X), np.array(y)

X, y = extract_window_data(input_data_normalized, target_data_normalized, window_len)


# Split the data into training and testing sets
split_ratio = 0.8  # Ratio of training set to total data
split_index = int(split_ratio * len(X))

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Creating model
model = build_lstm_model(X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss, optimizer=optimizer)

# Saved Weights
file_path = "./LSTM_" + ticker + "_weights.h5"

# Loads the weights
model.load_weights(file_path)

# Step 4: Make predictions
preds = model.predict(X_test)
y_test = y[split_index:]

# Normalize the target data
scaler = MinMaxScaler()
target_data_normalized = scaler.fit_transform(target_data)

# Inverse normalize the predictions
preds = preds.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
preds = scaler.inverse_transform(preds)
y_test = scaler.inverse_transform(y_test)

fig = px.line(x=model_data.index[-len(y_test):],
              y=[y_test.flatten(), preds.flatten()])
newnames = {'wide_variable_0':'Real Values', 'wide_variable_1': 'Predictions'}
fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                      legendgroup = newnames[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])))
fig.update_layout(
    xaxis_title="Date",
    yaxis_title=ticker+" Price",
    legend_title=" ")
st.write(fig)


# --- INFO BUBBLE

about_data, news = st.tabs(["About", "News"])

with about_data:
  # Candlestick
  raw_data = ticker_obj.history(start=start_date, end=end_date)
  fig = go.Figure(data=[go.Candlestick(x=raw_data.index,
                  open=raw_data['Open'],
                  high=raw_data['High'],
                  low=raw_data['Low'],
                  close=raw_data['Close'])])
  fig.update_layout(
                  title=ticker + " candlestick : Open, High, Low and Close",
                  yaxis_title=ticker + ' Price')
  st.plotly_chart(fig)

  # Table
  history_data = raw_data.copy()

  # Formating index Date
  history_data.index = pd.to_datetime(history_data.index, format='%Y-%m-%d %H:%M:%S').date
  history_data.index.name = "Date"
  history_data.sort_values(by='Date', ascending=False, inplace=True)
  st.write(history_data)


with news:
  sNews = StockNews(ticker, save_news=False)
  sNews_df = sNews.read_rss()

  # Showing most recent news
  for i in range(10):
    st.subheader(f"{i+1} - {sNews_df['title'][i]}")
    st.write(sNews_df['summary'][i])
    date_object = datetime.strptime(sNews_df['published'][i], '%a, %d %b %Y %H:%M:%S %z')
    st.write(f"_{date_object.strftime('%A')}, {date_object.date()}_")