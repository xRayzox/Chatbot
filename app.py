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

# Time interval selection
interval_options = ['1m', '2m', '5m', '15m', '30m', '60m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
selected_interval = st.sidebar.selectbox('Select Time Interval', interval_options, index=6)  # default: '1h'

start_date = st.sidebar.date_input('Start Date', date.today() - timedelta(days=365))
end_date = st.sidebar.date_input('End Date', date.today())

# --- MAIN PAGE
st.header('Cryptocurrency Prediction with Auto Candle Insights')

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

# --- DATA FETCHING using the selected interval
model_data = ticker_obj.history(interval=selected_interval, start=start_date, end=end_date)
if model_data.empty:
    st.error("No data fetched. Please check the date range, ticker, or selected interval.")
    st.stop()

# Calculate a 20-period moving average for charting
if 'Close' in model_data.columns:
    model_data['MA_20'] = model_data['Close'].rolling(window=20).mean()

# --- PREPARE DATA FOR MODELING
target_data = model_data["Close"].values.reshape(-1, 1)
scaler_target = MinMaxScaler()
target_data_normalized = scaler_target.fit_transform(target_data)

input_features = ['Open', 'High', 'Low', 'Volume']
input_data = model_data[input_features].values
scaler_features = MinMaxScaler()
input_data_normalized = scaler_features.fit_transform(input_data)

def build_lstm_model(input_data, output_size, neurons, activ_func='linear', dropout=0.2, loss='mse', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer)
    return model

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
split_index = int(split_ratio * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

model = build_lstm_model(X_train, output_size=1, neurons=lstm_neurons,
                         dropout=dropout_rate, loss=loss_function, optimizer=optimizer_type)

file_path = f"./LSTM_{ticker}_weights.h5"
if os.path.exists(file_path):
    model.load_weights(file_path)
else:
    st.warning(f"Weight file {file_path} not found. The model predictions may not be accurate.")

preds = model.predict(X_test)
preds = preds.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
preds_inverse = scaler_target.inverse_transform(preds)
y_test_inverse = scaler_target.inverse_transform(y_test)

fig = px.line(
    x=model_data.index[-len(y_test_inverse):],
    y=[y_test_inverse.flatten(), preds_inverse.flatten()],
    labels={'x': "Date", 'value': f"{ticker} Price"},
    title='Real Values vs Predictions'
)
fig.for_each_trace(lambda t: t.update(
    name='Real Values' if t.name == 'wide_variable_0' else 'Predictions',
    legendgroup='Real Values' if t.name == 'wide_variable_0' else 'Predictions'
))
fig.update_layout(legend_title="Legend")
st.write(fig)

# === AUTO CANDLE INSIGHTS FUNCTIONALITY ===
def detect_candlestick_patterns(prev, curr):
    """Detect common candlestick patterns between two consecutive candles."""
    patterns = []
    
    # Bullish Engulfing: previous candle bearish and current candle bullish, with body engulfing.
    if prev['Close'] < prev['Open'] and curr['Close'] > curr['Open']:
        if curr['Open'] < prev['Close'] and curr['Close'] > prev['Open']:
            patterns.append('Bullish Engulfing')
    
    # Bearish Engulfing: previous candle bullish and current candle bearish, with body engulfing.
    if prev['Close'] > prev['Open'] and curr['Close'] < curr['Open']:
        if curr['Open'] > prev['Close'] and curr['Close'] < prev['Open']:
            patterns.append('Bearish Engulfing')
    
    # Calculate candle components for the current candle
    body = abs(curr['Close'] - curr['Open'])
    lower_shadow = curr['Open'] - curr['Low'] if curr['Close'] >= curr['Open'] else curr['Close'] - curr['Low']
    upper_shadow = curr['High'] - curr['Close'] if curr['Close'] >= curr['Open'] else curr['High'] - curr['Open']
    
    # Hammer: small body near the top with a long lower shadow (at least twice the body) and little upper shadow.
    if body > 0 and lower_shadow > 2 * body and upper_shadow < body:
        patterns.append('Hammer')
    
    # Shooting Star: small body near the bottom with a long upper shadow (at least twice the body) and little lower shadow.
    if body > 0 and upper_shadow > 2 * body and lower_shadow < body:
        patterns.append('Shooting Star')
    
    return patterns

def generate_auto_insights(df):
    """
    Generate auto insights by scanning the candlestick data for patterns.
    Returns a list of strings with detected patterns and the corresponding date.
    """
    insights = []
    # Ensure the DataFrame has a Date column if not, reset_index will create one.
    df_insights = df.copy().reset_index()
    
    # Loop over candles starting from the second one (to compare with the previous candle)
    for i in range(1, len(df_insights)):
        prev = df_insights.loc[i-1]
        curr = df_insights.loc[i]
        patterns = detect_candlestick_patterns(prev, curr)
        if patterns:
            candle_date = curr['Date'] if 'Date' in curr else curr['index']
            insight = f"On {candle_date.date() if isinstance(candle_date, pd.Timestamp) else candle_date}: " \
                      + ", ".join(patterns) + " pattern detected."
            insights.append(insight)
    return insights

# Generate insights from the fetched data (using raw_data from history)
raw_data = ticker_obj.history(interval=selected_interval, start=start_date, end=end_date)
if not raw_data.empty:
    auto_insights = generate_auto_insights(raw_data)
else:
    auto_insights = ["No data available for insights."]

# --- TABS FOR ADDITIONAL INFO
tabs = st.tabs(["About", "News"])

with tabs[0]:
    st.subheader("Candlestick Chart with MA 20")
    # Candlestick chart with moving average
    if not raw_data.empty:
        raw_data['MA_20'] = raw_data['Close'].rolling(window=20).mean()
        fig2 = go.Figure(data=[go.Candlestick(
            x=raw_data.index,
            open=raw_data['Open'],
            high=raw_data['High'],
            low=raw_data['Low'],
            close=raw_data['Close'],
            name="OHLC"
        )])
        # Add MA trace
        fig2.add_trace(go.Scatter(
            x=raw_data.index,
            y=raw_data['MA_20'],
            mode='lines',
            name='MA 20'
        ))
        fig2.update_layout(
            title=f"{ticker} Candlestick Chart with MA 20",
            yaxis_title=f"{ticker} Price"
        )
        st.plotly_chart(fig2)
        
        st.subheader("Historical Data")
        history_data = raw_data.copy()
        history_data.index = pd.to_datetime(history_data.index).date
        history_data.index.name = "Date"
        history_data.sort_index(ascending=False, inplace=True)
        st.write(history_data)
        
        st.subheader("Auto Candle Insights")
        if auto_insights:
            for insight in auto_insights[-5:]:  # show last 5 insights
                st.info(insight)
        else:
            st.info("No candlestick patterns detected in the recent data.")
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
