# Keras Real-Time Crypto Forecasting

This project is a Streamlit-based web application that forecasts cryptocurrency prices using LSTM models built with Keras. The app provides real-time price data and predictions for several cryptocurrencies, including Bitcoin (BTC), Ethereum (ETH), YFI, CORE, and XAUT.

## Features

- **Real-Time Price Tracking**: Displays current and historical price data for selected cryptocurrencies.
- **Price Predictions**: Forecasts prices for the next day, week, and month using LSTM models.
- **Historical Price Visualization**: Interactive charts showing historical price data with selectable time intervals.

## Live Demo

You can view the live demo of the application [here](https://keras-realtime-crypto-forecasting.streamlit.app/).

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/keras-realtime-crypto-forecasting.git
    cd keras-realtime-crypto-forecasting
    ```

2. **Create a virtual environment and activate it**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Streamlit app**:
    ```bash
    streamlit run crypto.py
    ```

## Dependencies

The required Python packages for this project are listed in `requirements.txt`. They include:

- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`
- `tensorflow`
- `requests`
- `plotly`

## Usage

1. Open the application in your browser.
2. Select a cryptocurrency from the dropdown menu to view real-time prices and predictions.
3. Choose a time interval to see historical price data.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please fork the repository and submit a pull request. 

## License

This project is licensed under the MIT License.

## Acknowledgments

- [CoinGecko API](https://coingecko.com) for cryptocurrency data.
- [Streamlit](https://streamlit.io) for creating interactive web applications.
- [Keras](https://keras.io) and [TensorFlow](https://tensorflow.org) for machine learning models.
