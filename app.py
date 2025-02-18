import os
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
end_date = st.sidebar.date_input('End Date', date.today())

# --- MAIN PAGE
st.header('Cryptocurrency Prediction')

col1, col2 = st.columns([1, 9])
with col1:
    icon_path = f'icons/{ticker}.png'
    if os.path.exists(icon_path):
        st.image(icon_path, width=75)
    else:
        st.write("No icon available")
with col2:
    st.write(f"## {ticker}")

ticker_obj = yf.Ticker(ticker)

# --- DATA FETCHING
model_data = ticker_obj.history(interval='1h', start=start_date, end=end_date)
if model_data.empty:
    st.error("No data fetched. Please check the date range or ticker.")
    st.stop()

# Extract the 'Close' column for prediction
target_data = model_data["Close"].values.reshape(-1, 1)

# Normalize the target data using its own scaler
scaler_target = MinMaxScaler()
target_data_normalized = scaler_target.fit_transform(target_data)

# Normalize the input features using a separate scaler
input_features = ['Open', 'High', 'Low', 'Volume']
input_data = model_data[input_features].values
scaler_features = MinMaxScaler()
input_data_normalized = scaler_features.fit_transform(input_data)

# --- MODEL BUILDING
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
split_ratio = 0.8
lstm_neurons = 50
epochs = 100
batch_size = 128
loss_function = 'mean_squared_error'
dropout_rate = 0.24
optimizer_type = 'adam'

def extract_window_data(input_data, target_data, window_len):
    X, y = [], []
    for i in range(len(input_data) - window_len):
        X.append(input_data[i: i + window_len])
        y.append(target_data[i + window_len])
    return np.array(X), np.array(y)

X, y = extract_window_data(input_data_normalized, target_data_normalized, window_len)

# Split the data into training and testing sets
split_index = int(split_ratio * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Create and load the model
model = build_lstm_model(X_train, output_size=1, neurons=lstm_neurons,
                         dropout=dropout_rate, loss=loss_function, optimizer=optimizer_type)

file_path = f"./LSTM_{ticker}_weights.h5"
if os.path.exists(file_path):
    model.load_weights(file_path)
else:
    st.warning(f"Weight file {file_path} not found. The model predictions may not be accurate.")

# --- PREDICTIONS
preds = model.predict(X_test)

# Inverse transform predictions and actual values using the target scaler
preds = preds.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
preds_inverse = scaler_target.inverse_transform(preds)
y_test_inverse = scaler_target.inverse_transform(y_test)

# Plot predictions vs. real values
fig = px.line(
    x=model_data.index[-len(y_test_inverse):],
    y=[y_test_inverse.flatten(), preds_inverse.flatten()],
    labels={'x': "Date", 'value': f"{ticker} Price"},
    title='Real Values vs Predictions'
)
# Rename traces (default names are "wide_variable_0" and "wide_variable_1")
fig.for_each_trace(lambda t: t.update(
    name='Real Values' if t.name == 'wide_variable_0' else 'Predictions',
    legendgroup='Real Values' if t.name == 'wide_variable_0' else 'Predictions'
))
fig.update_layout(legend_title="Legend")
st.write(fig)

# --- TABS FOR ADDITIONAL INFO
tabs = st.tabs(["About", "News"])

with tabs[0]:
    # Candlestick chart
    raw_data = ticker_obj.history(start=start_date, end=end_date)
    if not raw_data.empty:
        fig2 = go.Figure(data=[go.Candlestick(
            x=raw_data.index,
            open=raw_data['Open'],
            high=raw_data['High'],
            low=raw_data['Low'],
            close=raw_data['Close']
        )])
        fig2.update_layout(
            title=f"{ticker} Candlestick: Open, High, Low and Close",
            yaxis_title=f"{ticker} Price"
        )
        st.plotly_chart(fig2)

        # Display historical data in a table
        history_data = raw_data.copy()
        history_data.index = pd.to_datetime(history_data.index).date
        history_data.index.name = "Date"
        history_data.sort_index(ascending=False, inplace=True)
        st.write(history_data)
    else:
        st.error("No historical data available for candlestick chart.")

with tabs[1]:
    try:
        sNews = StockNews(ticker, save_news=False)
        sNews_df = sNews.read_rss()
        if sNews_df.empty:
            st.info("No news available.")
        else:
            num_news = min(10, len(sNews_df))
            for i in range(num_news):
                st.subheader(f"{i+1} - {sNews_df['title'][i]}")
                st.write(sNews_df['summary'][i])
                date_object = datetime.strptime(sNews_df['published'][i], '%a, %d %b %Y %H:%M:%S %z')
                st.write(f"_{date_object.strftime('%A')}, {date_object.date()}_")
    except Exception as e:
        st.error(f"Error fetching news: {e}")
