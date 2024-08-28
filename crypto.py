import pandas as pd
import requests
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import streamlit as st
import plotly.graph_objs as go
from datetime import datetime

# Function to obtain historical data
def get_historical_prices(crypto_id, vs_currency='mxn', days=365):
    url = f'https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart'
    params = {
        'vs_currency': vs_currency,
        'days': days,
        'interval': 'daily'
    }
    response = requests.get(url, params=params)
    data = response.json()
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Obtain historical data from last 365 days for cryptocurrencies
bitcoin_prices = get_historical_prices('bitcoin', days=365)
ethereum_prices = get_historical_prices('ethereum', days=365)
yfi_prices = get_historical_prices('yearn-finance', days=365)
core_prices = get_historical_prices('coredao', days=365)
xaut_prices = get_historical_prices('tether-gold', days=365)

# Preprocess data
def preprocess_data(data, seq_length):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i+seq_length])
        y.append(scaled_data[i+seq_length])
    return np.array(X), np.array(y), scaler

# Set number of days for prediction
seq_length = 60

bitcoin_prices_values = bitcoin_prices['price'].values
ethereum_prices_values = ethereum_prices['price'].values
yfi_prices_values = yfi_prices['price'].values
core_prices_values = core_prices['price'].values
xaut_prices_values = xaut_prices['price'].values

X_bitcoin, y_bitcoin, scaler_bitcoin = preprocess_data(bitcoin_prices_values, seq_length)
X_ethereum, y_ethereum, scaler_ethereum = preprocess_data(ethereum_prices_values, seq_length)
X_yfi, y_yfi, scaler_yfi = preprocess_data(yfi_prices_values, seq_length)
X_core, y_core, scaler_core = preprocess_data(core_prices_values, seq_length)
X_xaut, y_xaut, scaler_xaut = preprocess_data(xaut_prices_values, seq_length)

# Create y train model
def create_model(seq_length):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model_bitcoin = create_model(seq_length)
model_bitcoin.fit(X_bitcoin, y_bitcoin, epochs=20, batch_size=32, verbose=1)

model_ethereum = create_model(seq_length)
model_ethereum.fit(X_ethereum, y_ethereum, epochs=20, batch_size=32, verbose=1)

model_yfi = create_model(seq_length)
model_yfi.fit(X_yfi, y_yfi, epochs=20, batch_size=32)

model_core = create_model(seq_length)
model_core.fit(X_core, y_core, epochs=20, batch_size=32)

model_xaut = create_model(seq_length)
model_xaut.fit(X_xaut, y_xaut, epochs=20, batch_size=32)

# Function for predictions
def make_predictions(model, data, scaler, days=1):
    predictions = []
    input_seq = data.reshape(1, seq_length, 1)
    for _ in range(days):
        predicted_price = model.predict(input_seq)
        predictions.append(predicted_price[0][0])
        predicted_price = np.reshape(predicted_price, (1, 1, 1))
        input_seq = np.append(input_seq[:, 1:, :], predicted_price, axis=1)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

# Predictionsfor: 
# Bitcoin
next_day_bitcoin = make_predictions(model_bitcoin, X_bitcoin[-1], scaler_bitcoin, days=1)
next_week_bitcoin = make_predictions(model_bitcoin, X_bitcoin[-1], scaler_bitcoin, days=7)
next_month_bitcoin = make_predictions(model_bitcoin, X_bitcoin[-1], scaler_bitcoin, days=30)

# Ethereum
next_day_ethereum = make_predictions(model_ethereum, X_ethereum[-1], scaler_ethereum, days=1)
next_week_ethereum = make_predictions(model_ethereum, X_ethereum[-1], scaler_ethereum, days=7)
next_month_ethereum = make_predictions(model_ethereum, X_ethereum[-1], scaler_ethereum, days=30)

# YFI
next_day_yfi = make_predictions(model_yfi, X_yfi[-1], scaler_yfi, days=1)
next_week_yfi = make_predictions(model_yfi, X_yfi[-1], scaler_yfi, days=7)
next_month_yfi = make_predictions(model_yfi, X_yfi[-1], scaler_yfi, days=30)

# CORE
next_day_core = make_predictions(model_core, X_core[-1], scaler_core, days=1)
next_week_core = make_predictions(model_core, X_core[-1], scaler_core, days=7)
next_month_core = make_predictions(model_core, X_core[-1], scaler_core, days=30)

# XAUT
next_day_xaut = make_predictions(model_xaut, X_xaut[-1], scaler_xaut, days=1)
next_week_xaut = make_predictions(model_xaut, X_xaut[-1], scaler_xaut, days=7)
next_month_xaut = make_predictions(model_xaut, X_xaut[-1], scaler_xaut, days=30)

# Configurate Streamlit
st.title('BTC & ETH - Price + Prediction')

# Criptocurrency selector
crypto_option = st.selectbox(
    'Select Cryptocurrency:',
    ['Bitcoin', 'Ethereum', 'YFI', 'CORE', 'XAUT']
)

# Show actual price of selected crypto
if crypto_option == 'Bitcoin':
    current_price = bitcoin_prices['price'].values[-1]
    prev_price = bitcoin_prices['price'].values[-2]
elif crypto_option == 'Ethereum':
    current_price = ethereum_prices['price'].values[-1]
    prev_price = ethereum_prices['price'].values[-2]
elif crypto_option == 'YFI':
    current_price = yfi_prices['price'].values[-1]
    prev_price = yfi_prices['price'].values[-2]
elif crypto_option == 'CORE':
    current_price = core_prices['price'].values[-1]
    prev_price = core_prices['price'].values[-2]
else:  # XAUT
    current_price = xaut_prices['price'].values[-1]
    prev_price = xaut_prices['price'].values[-2]

# Determine icon of value change
if current_price > prev_price:
    price_change_icon = """
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 2L15 8H9L12 2Z" fill="green"/>
    </svg>
    """
    icon_color = 'green'
else:
    price_change_icon = """
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 22L15 16H9L12 22Z" fill="red"/>
    </svg>
    """
    icon_color = 'red'

# Show actual price and change icon
st.subheader('Actual Price')
st.markdown(f"{crypto_option}: ${current_price:.2f} MXN {price_change_icon}", unsafe_allow_html=True)

# Show predictions
st.subheader('Predictions')
if crypto_option == 'Bitcoin':
    st.write(f"Next Day: ${next_day_bitcoin[0][0]:.2f} MXN")
    st.write(f"Next Week: ${next_week_bitcoin[-1][0]:.2f} MXN")
    st.write(f"Next Month: ${next_month_bitcoin[-1][0]:.2f} MXN")
elif crypto_option == 'Ethereum':
    st.write(f"Next Day: ${next_day_ethereum[0][0]:.2f} MXN")
    st.write(f"Next Week: ${next_week_ethereum[-1][0]:.2f} MXN")
    st.write(f"Next Month: ${next_month_ethereum[-1][0]:.2f} MXN")
elif crypto_option == 'YFI':
    st.write(f"Next Day: ${next_day_yfi[0][0]:.2f} MXN")
    st.write(f"Next Week: ${next_week_yfi[-1][0]:.2f} MXN")
    st.write(f"Next Month: ${next_month_yfi[-1][0]:.2f} MXN")
elif crypto_option == 'CORE':
    st.write(f"Next Day: ${next_day_core[0][0]:.2f} MXN")
    st.write(f"Next Week: ${next_week_core[-1][0]:.2f} MXN")
    st.write(f"Next Month: ${next_month_core[-1][0]:.2f} MXN")
elif crypto_option == 'XAUT':
    st.write(f"Next Day: ${next_day_xaut[0][0]:.2f} MXN")
    st.write(f"Next Week: ${next_week_xaut[-1][0]:.2f} MXN")
    st.write(f"Next Month: ${next_month_xaut[-1][0]:.2f} MXN")

# Interactive History Chart
st.subheader('Historical Prices')

# Interval selector
interval_option = st.selectbox(
    'Select Time Interval:',
    ['1d', '1w', '1m', '1y']
)

# Convert intervals to frequencies
interval_dict = {
    '1d': 'D',
    '1w': 'W',
    '1m': 'M',
    '1y': 'A'
}
freq = interval_dict[interval_option]

def filter_and_resample(df, freq):
    df = df.set_index('timestamp')
    resampled_df = df.resample(freq).mean()
    return resampled_df

# Filter and resample data
if crypto_option == 'Bitcoin':
    filtered_data = filter_and_resample(bitcoin_prices, freq)
elif crypto_option == 'Ethereum':
    filtered_data = filter_and_resample(ethereum_prices, freq)
elif crypto_option == 'YFI':
    filtered_data = filter_and_resample(yfi_prices, freq)
elif crypto_option == 'CORE':
    filtered_data = filter_and_resample(core_prices, freq)
elif crypto_option == 'XAUT':
    filtered_data = filter_and_resample(xaut_prices, freq)

# Plotting historical prices
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=filtered_data.index,
    y=filtered_data['price'],
    mode='lines',
    name=crypto_option
))

fig.update_layout(
    title=f'Price History of {crypto_option}',
    xaxis_title='Date',
    yaxis_title='Price (MXN)',
    xaxis_rangeslider_visible=True
)

st.plotly_chart(fig)