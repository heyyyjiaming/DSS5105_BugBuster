import pandas as pd
import numpy as np
import plotly.express as px
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import keras




########################################### Stock ###########################################
def stock_data_manipulation(stock_price):
    stock_price.dropna(inplace = True)
    stock_price['Date'] = pd.to_datetime(stock_price['Date'], format = '%Y-%m-%d')
    stock_price['Adj Close'] = pd.to_numeric(stock_price['Adj Close'], errors = 'coerce')
    stock_price['Log Return'] = np.log(stock_price['Adj Close'] / stock_price['Adj Close'].shift(1))
    stock_price.dropna(inplace = True)
    stock_price.set_index('Date', inplace = True)
    
    return stock_price



def rolling_vol_plot(stock_price):
    # Calculate rolling volatility (standard deviation of log returns)
    window = 30  # 30-day rolling window
    stock_price['Rolling Volatility'] = stock_price['Log Return'].rolling(window=window).std()

    fig_volatility_risk = px.line(stock_price, x=stock_price.index, y='Rolling Volatility', 
                                  title='Rolling Volatility (Risk) Over Time',
                                  labels={'Date': 'Date', 'Rolling Volatility': 'Volatility'})
    return fig_volatility_risk



def volatility_pred(stock_price):
    # Build GARCH model
    am = arch_model(stock_price['Log Return']*100, vol='GARCH', p=1, q=1)
    res = am.fit(disp='off')

    # Predict the volatility of next 90 days
    forecast = res.forecast(horizon=90)

    # Extract the variance of the prediction
    variance = forecast.variance.values[-1,:]

    # Calculation of conditional standard deviation (volatility)
    cond_vol = np.sqrt(variance)

    # Create date index
    forecast_index = pd.date_range(start=stock_price.index[-1], periods=90, freq='D')

    # Assuming forecast_index and cond_vol are defined as in the previous context
    fig_volatility_pred = px.line(x=forecast_index, y=cond_vol, labels={'x': 'Date', 'y': 'Conditional volatility'}, 
                                  title='The next 90 day volatility predicted by the GARCH model')
    return fig_volatility_pred


def compute_RSI(data, time_window):
    """
    Calculate RSI

    Parameters:
    - data: pandas Series, corresponding to price data (such as closing price).
    - time_window: specifies the time window for calculating the RSI (usually 14).

    Return：
    - rsi: pandas Series
    """
    diff = data.diff(1).dropna()  # Calculate Price chancge
    up_chg = 0 * diff  # Initialize the increase sequence
    down_chg = 0 * diff  # Initialize the drop sequence

    # Separate the gains from the losses
    up_chg[diff > 0] = diff[diff > 0]
    down_chg[diff < 0] = -diff[diff < 0]

    # Calculate the average increase and average decrease
    up_chg_avg = up_chg.rolling(window=time_window, min_periods=time_window).mean()
    down_chg_avg = down_chg.rolling(window=time_window, min_periods=time_window).mean()

    # Handles the first average to avoid NaN values
    up_chg_avg = up_chg_avg.fillna(value=up_chg.expanding().mean())
    down_chg_avg = down_chg_avg.fillna(value=down_chg.expanding().mean())

    # Calculate RSI
    rs = up_chg_avg / down_chg_avg
    rsi = 100 - 100 / (1 + rs)
    return rsi


# Anti-normalization
def inverse_transform(pred, scaler, n_features):
    expanded = np.concatenate([pred, np.zeros((pred.shape[0], n_features - 1))], axis=1)
    return scaler.inverse_transform(expanded)[:, 0]


def stock_pred_model(stock_price, time_step):
    # Add technical specifications (Add more features)
    stock_price['MA10'] = stock_price['Adj Close'].rolling(window=10).mean()
    stock_price['MA20'] = stock_price['Adj Close'].rolling(window=20).mean()
    stock_price['MA50'] = stock_price['Adj Close'].rolling(window=50).mean()
    stock_price['EMA10'] = stock_price['Adj Close'].ewm(span=10, adjust=False).mean()
    stock_price['RSI'] = compute_RSI(stock_price['Adj Close'], 14)  # 需要定义compute_RSI函数
    stock_price.dropna(inplace=True)
    
    # Prepare Data
    features = ['Adj Close', 'MA10', 'MA20', 'MA50', 'EMA10', 'RSI']
    data = stock_price[features].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Create a multi-feature dataset
    def create_dataset_multifeature(data, time_step=60):
        X, Y = [], []
        for i in range(time_step, len(data)):
            X.append(data[i - time_step:i])
            Y.append(data[i, 0])  # The target variable is 'Adj Close'
            
        return np.array(X), np.array(Y)
        
    X, Y = create_dataset_multifeature(scaled_data, time_step)
    
    # Split data set
    train_size = int(len(X) * 0.8)
    X_train, X_valid = X[:train_size], X[train_size:]
    Y_train, Y_valid = Y[:train_size], Y[train_size:]
    
    # Build an improved LSTM model
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    
    # Compile the model, using the learning rate scheduler
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    # Set the callback function
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    
    # Training model
    history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid),
                    epochs=200, batch_size=64, callbacks=[early_stop, reduce_lr], verbose=1)
    
    # Make predictions on test sets
    predictions = model.predict(X_valid)
    

    predictions = inverse_transform(predictions, scaler, len(features))
    Y_valid_actual = inverse_transform(Y_valid.reshape(-1, 1), scaler, len(features))
    
    return stock_price, scaled_data, model, features, scaler # train_size, Y_valid_actual, predictions
    
    
    
def stock_pred(stock_price, scaled_data, features, model, time_step, scaler):
    last_60_days = scaled_data[-time_step:]
    future_predictions = []
    
    for _ in range(90):
        input_data = last_60_days.reshape(1, time_step, len(features))
        predicted_value = model.predict(input_data)
        future_predictions.append(predicted_value[0, 0])
        
        # Update last_60_days to add a new predicted value and remove the earliest time step
        last_60_days = np.vstack([last_60_days[1:], np.concatenate([predicted_value[0], last_60_days[-1, 1:]])])
    
    # Reverse normalize the predicted value of the future
    future_predictions = inverse_transform(np.array(future_predictions).reshape(-1, 1), scaler, len(features))

    # Create future date
    last_date = pd.to_datetime(stock_price.index[-1])
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=90)

    # Create a future prediction data box
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Price': future_predictions
        })
    
    # Plot the prediction
    fig = px.line(future_df, x='Date', y='Predicted Price',
                  title='Future Stock Price Predictions')
    fig.update_layout(xaxis_title='Date', yaxis_title='Predicted Close Price')
    
    return fig, future_df
    
    
    
def var_calculate(future_df, confidence_level):
    
    # Calculate daily forecast returns
    future_df['Predicted Return'] = future_df['Predicted Price'].pct_change()
    
    # Remove NaN value (because the first yield cannot be calculated)
    future_returns = future_df['Predicted Return'].dropna()
    
    # Calculate VaR
    VaR = np.percentile(-future_returns, confidence_level * 100)
    
    # Calculate CVaR
    CVaR = -future_returns[future_returns <= -VaR].mean()
    
    # Plot VaR and CVaR
    loss_df = pd.DataFrame({'Loss': -future_returns})
    
    # Create a histogram of the loss distribution
    fig = px.histogram(loss_df, x='Loss', nbins=30, opacity=0.7, title='Predict the loss distribution of returns')
    
    fig.add_vline(x=VaR, line_dash="dash", line_color="red",
                  annotation_text=f"VaR ({VaR:.4f})", annotation_position="top left")
    
    fig.add_vline(x=CVaR, line_dash="dash", line_color="green",
                  annotation_text=f"CVaR ({CVaR:.4f})", annotation_position="top right")
    
    fig.update_layout(
        xaxis_title='Loss',
        yaxis_title='Frequency'
        )

    return VaR, CVaR, fig
    
    
