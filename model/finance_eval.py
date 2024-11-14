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
    fig_volatility_risk.add_hline(y=0.025, line_dash="dash", line_color="red", line_width=2)

    return stock_price['Rolling Volatility'], fig_volatility_risk



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



def volatility_analysis_invest(volatility):
    volatility_invest_text = {'Low':'''The historical data shows volatility has stabilized at a lower level, suggesting a more stable and less risky market environment. 
                     Investors may consider increasing exposure to growth stocks or other higher-risk assets, capitalizing on the reduced uncertainty. 
                     However, maintaining a diversified portfolio remains essential to mitigate unexpected market shocks.''', 
                     
                     'High': '''The historical data shows volatility has stabilized at a lower level, suggesting a more stable and less risky market environment. 
                     Investors may consider increasing exposure to growth stocks or other higher-risk assets, capitalizing on the reduced uncertainty. 
                     However, maintaining a diversified portfolio remains essential to mitigate unexpected market shocks.'''}
    volatility.dropna(inplace = True)
    high_vol = len([x for x in volatility if x > 0.025])
    
    if high_vol <= len(volatility)/3:
        volatility_invest = volatility_invest_text['Low']
    else:
        volatility_invest = volatility_invest_text['High']
    
    return volatility_invest






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





############################### Finance Analysis ########################################
def fin_data_manipulate(financial_database, fin_sub_mean, fin_mean, company_name):
    sub_sectors = {'Software and Services': ['Captii','CSE Global','V2Y Corp','SinoCloud Grp'],
               'Technology Hardware and Equipment': ['Addvalue Tech','Nanofilm','Venture'],
               'Semiconductors and Semiconductor Equipment': ['AdvancedSystems','AEM SGD','Asia Vets','ASTI','UMS',],
               'Information Technology':['Audience'], 
               'Engineering Services': ['ST Engineering','Singtel','GSS Energy']}
    
    fin_mean.insert(1, 'Type', 'Technology Industry')
    
    company_fin = financial_database[financial_database['Company'].str.lower() == company_name.lower()].reset_index(drop=True)
    company_fin = company_fin[[company_fin.columns[-2], company_fin.columns[-1]] + list(company_fin.columns[:-2])]
    company_fin.rename(columns = {'Company' : 'Type'}, inplace = True)
    
    for sector, companies in sub_sectors.items():
        if company_name.lower() in [company.lower() for company in companies]:
            sub_sector = sector
    
    sub_fin = fin_sub_mean[fin_sub_mean['Sub-sector'] == sub_sector]
    sub_fin.rename(columns = {'Sub-sector' : 'Type'}, inplace = True)
    
    fin_df = pd.concat([fin_mean, sub_fin, company_fin], ignore_index=True)
    
    return fin_df
    
    
    # Define the function to plot the graphs and return the plot objects
def plot_financial_data(fin_df):
    figures = []  # List to store the plotly figures
    # Loop through each column starting from the 3rd column (index 2)
    for col in fin_df.columns[2:].drop('Diluted Normalised EPS Growth (%)'):
        fig = px.line(fin_df, x='Year', y=col, color='Type',
                      title=f'Financial Data for {col}',
                      labels={'Year': 'Year', col: col})
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
        figures.append(fig)  # Store the plot in the list
    return figures



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


# risk_pref = 'Low'
# VaR_text = {'Low':'''The VaR is below 0.01, it indicates a low level of risk exposure, meaning that the portfolio is very unlikely to incur significant losses within the specified time frame.
#         If the current low risk aligns with your risk tolerance, maintaining the portfolio as it is could be a prudent choice.
#         If your target returns are not being met, consider increasing risk exposure by diversifying into higher-risk, higher-return assets.
#         Both remember to maintain your portfolio as well-diversified to mitigate any unforeseen risks''', 
        
#         'Middel': '''The VaR is between 0.01 and 0.05, the portfolio is exposed to a moderate level of risk. 
#         This range suggests that there is a reasonable balance between risk and potential return.
#         Compare the current risk-return tradeoff with your investment goals. If returns are satisfactory, you may maintain the current portfolio.
#         If you believe the risk is high even with VaR between 0.01 and 0.05, you can Reallocate a portion of your portfolio to low-risk investments such as government bonds, blue-chip stocks, or money market funds.''',
        
#         'High':'''The VaR exceeds 0.05, it indicates a higher risk exposure, meaning the portfolio is more likely to experience significant losses within the specified time frame. 
#         In such a scenario, it's crucial to carefully manage risk while considering potential opportunities for higher returns.
#         If the current level of risk is above your comfort zone, consider adjusting the portfolio. You can shift some capital to lower-risk assets like bonds, defensive stocks, or other safe-haven investments.
#         If the current level of risk is acceptable, you may consider maintaining your current strategy while optimizing for potential returns. 
#         Remember to keep a close eye on the portfolio’s performance and market trends to ensure the risk remains manageable. 
#         Even if the risk is acceptable, conduct stress tests to anticipate portfolio behavior under extreme scenarios.'''}
def risk_analysis(risk_pref, VaR):
    VaR_text = {'Low':'''The VaR is below 0.01, it indicates a low level of risk exposure, meaning that the portfolio is very unlikely to incur significant losses within the specified time frame.
        If the current low risk aligns with your risk tolerance, maintaining the portfolio as it is could be a prudent choice.
        If your target returns are not being met, consider increasing risk exposure by diversifying into higher-risk, higher-return assets.
        Both remember to maintain your portfolio as well-diversified to mitigate any unforeseen risks''', 
        
        'Middel': '''The VaR is between 0.01 and 0.05, the portfolio is exposed to a moderate level of risk. 
        This range suggests that there is a reasonable balance between risk and potential return.
        Compare the current risk-return tradeoff with your investment goals. If returns are satisfactory, you may maintain the current portfolio.
        If you believe the risk is high even with VaR between 0.01 and 0.05, you can Reallocate a portion of your portfolio to low-risk investments such as government bonds, blue-chip stocks, or money market funds.''',
        
        'High':'''The VaR exceeds 0.05, it indicates a higher risk exposure, meaning the portfolio is more likely to experience significant losses within the specified time frame. 
        In such a scenario, it's crucial to carefully manage risk while considering potential opportunities for higher returns.
        If the current level of risk is above your comfort zone, consider adjusting the portfolio. You can shift some capital to lower-risk assets like bonds, defensive stocks, or other safe-haven investments.
        If the current level of risk is acceptable, you may consider maintaining your current strategy while optimizing for potential returns. 
        Remember to keep a close eye on the portfolio’s performance and market trends to ensure the risk remains manageable. 
        Even if the risk is acceptable, conduct stress tests to anticipate portfolio behavior under extreme scenarios.'''}
    if VaR <= 0.01:
        VaR_con = 'Low'
        VaR_analysis = VaR_text['Low']
    elif VaR >= 0.05:
        VaR_con = 'High'
        VaR_analysis = VaR_text['High']
    else:
        VaR_con = 'Middle'
        VaR_analysis = VaR_text['Middle']
        
    if risk_pref == VaR_con:
        match_con = 'The predicted risk matches with your risk preference level.'
    else:
        match_con = 'The predicted risk does not match your risk preference level.'
        
    return match_con, VaR_analysis
    



    
    
    
############################################## Overall Financial Analysis ##############################################
def investor_analyze_financial_metrics(company_data, industry_data):

    # 清理数据，确保列名一致，年份为数值类型
    company_data.columns = company_data.columns.str.strip()
    company_data['Year'] = pd.to_numeric(company_data['Year'], errors='coerce')

    # 筛选2023和2024年的数据
    metrics_comparison = company_data[company_data['Year'].isin([2023, 2024])][
        ['Year', 'Return on Equity (ROE)', 'Return on Assets (ROA)', 'Operating Margin', 'Net Profit Margin', 
         'Diluted Normalised EPS', 'Current Ratio', 'Asset Turnover', 'Quick Ratio']
    ]

    # 提取2024年的industry average
    industry_metrics = industry_data[industry_data['Year'] == 2024].iloc[0]

    # 提取2023年和2024年的指标数据
    year_metrics = metrics_comparison.set_index('Year').T

    # 结果生成函数
    def generate_result(metric, value_23, value_24, industry_value):
        trend_result = f"The company's {metric} is {'increasing' if value_24 > value_23 else 'decreasing'} from 2022 to 2023, "
        industry_comparison = f" and the company's {metric} is {'above' if value_24 > industry_value else 'below'} the industry average."

        # trend_result = f"The company's {metric} is {'increasing' if value_24 > value_23 else 'decreasing'}, indicating "
        # if metric == 'Return on Equity (ROE)':
        #     trend_result += "profitability changes per dollar of shareholders' money."
        # elif metric == 'Return on Assets (ROA)':
        #     trend_result += "efficiency in utilizing its assets."
        # elif metric == 'Operating Margin':
        #     trend_result += "profitability from its core business relative to revenue."
        # elif metric == 'Net Profit Margin':
        #     trend_result += "effectiveness in converting revenue to profit."
        # elif metric == 'Diluted Normalised EPS':
        #     trend_result += "changes in shareholder payouts."
        # elif metric == 'Current Ratio':
        #     trend_result += "changes in short-term liquidity."
        # elif metric == 'Quick Ratio':
        #     trend_result += "changes in immediate financial stability."
        # elif metric == 'Asset Turnover':
        #     trend_result += "efficiency in using its assets to generate revenue."
        
        # industry_comparison = f" The company's {metric} is {'above' if value_24 > industry_value else 'below'} the industry average."
        return trend_result + industry_comparison
    
    def generate_recommendation(metric, value_23, value_24, industry_value):
        if metric == 'Return on Equity (ROE)':
            Investment_Recommendations = "Investors can use ROE to estimate a stock’s growth rate and the growth rate of its dividends by comparing with the industrial ROE. A larger ROE indicates a better profitability and may suggest an excepted corporate growth. However, an extremely high ROE can also be the result of a small equity account compared to net income, which indicates risk. Publicly listed companies with a long-term ROE of 10% to 20% are considered excellent companies. However, an ROE below 10% does not necessarily indicate poor company performance, depending on the stage of development of the company."
            if value_24 > value_23:
                trend_rec = (
                        "An increased ROE means the company is making more profit from each dollar of shareholders' money. But it’s important to check what’s driving the increase, like higher profits or more debt.")  
            else:
                trend_rec = (
                            "A decreased ROE means the company is earning less profit from each dollar of shareholders' equity, indicating a declining profitability, inefficient use of equity, or increased costs.")  
            if value_24 > industry_value:
                industry_rec = ("The ROE of certain firm is higher than the industrial average means the company is more efficient in earning profits from its equity compared to its peers, reflecting a better management, stronger profitability, or more effective use of resources, which is a positive sign of investment.")  
            else: 
                industry_rec = ("The ROE of certain firm is lower than the industrial average means the company is less efficient in generating returns on its equity relative to competitors, reflecting a weak profitability, higher costs, or inefficient use of equity, which is a negative sign of investment.")
            return f"{trend_rec} + {industry_rec} + {Investment_Recommendations}"
        
        elif metric == 'Return on Assets (ROA)':
            Investment_Recommendations = "Investors can use ROA to find stock opportunities because the ROA shows how efficient a company is at using its assets to make profits. A higher ROA means a company is more efficient and productive at managing its balance sheet to generate profits. A lower ROA indicates the company might have over-invested in assets that have failed to produce revenue growth. The higher ROA is better, the better. However, it’s important not to rely solely on a single year’s data; instead, focusing on long-term trends."
            if value_24 > value_23:
                trend_rec = (
                        "An increased ROA means the company is earning more profit from each dollar of its assets, showing that the company is managing its resources effectively.")  
            else:
                trend_rec = (
                            "An increased ROA means the company is earning less profit from each dollar of its assets, indicating a declining profitability, increased costs, or underutilized assets")              
            if value_24 > industry_value:
                industry_rec = ("The ROA of certain firm is higher than the industrial average means the firm is more efficient at using its assets to make profits compared to its industry peers, reflecting a better operational efficiency, cost management, or revenue generation.")              
            else: 
                industry_rec = ("The ROA of certain firm is lower than the industrial average means the firm is less efficient at using its assets to make profits compared to its industry peers, reflecting a poor asset management, higher operational costs, or weaker revenue generation.")
            return f"{trend_rec} + {industry_rec} + {Investment_Recommendations} "
        
        
        elif metric == 'Operating Margin':
            Investment_Recommendations = "Investors can use Operating Profit Margin to see if a company is making profit primarily from its core operations or from other means, such as investing. It is one of the best ways to evaluate a company's operational efficiency. Rising operating margins show a company that is managing its costs and increasing its profits. Margins above the industry average or the overall market indicate financial efficiency and stability. However, margins below the industry average might indicate potential financial vulnerability to an economic downturn or financial distress if a trend develops. Investors can use Operating Profit Margin to predict the company’s trends in growth and to pinpoint unnecessary expenses. Also, investors can compare it to other companies within the same industry. The company owns higher margin is more worth investing."
            if value_24 > value_23:
                trend_rec = (
                        "An increased Operating Profit Margin means a company makes more profit from its core business in relation to its total revenue, indicating a strong operational performance which can support expansion or reinvestment.")  
            else:
                trend_rec = (
                            "A decreased Operating Profit Margin means a company makes less profit from its core business in relation to its total revenues, indicating operational inefficiencies, pricing pressures, or increased competition.")  
            if value_24 > industry_value:
                industry_rec = ("The Operating Profit Margin of certain firm is higher than the industrial average means the firm is more efficient at managing its operating costs and generating profit from its core operations compared to its peers, suggesting the firm has a robust and profitable business model, making it attractive to investors.")             
            else: 
                industry_rec = ("The Operating Profit Margin of certain firm is lower than the industrial average means the firm is less efficient at managing its operating costs and generating profit from its core operations compared to its peers, indicating higher operating costs, weaker pricing power, or operational inefficiencies.")
            return f"{trend_rec} + {industry_rec} + {Investment_Recommendations}"
 
        elif metric == 'Net Profit Margin':
            Investment_Recommendations = "Investors can use Net Profit Margin to assess if a company’s management is generating enough profit from its sales and whether operating costs and overhead costs are being contained. Generally, a track record of expanding margins mean that the net profit margin is rising over time. Meanwhile, companies that can expand their net margins over time are generally rewarded with share price growth, as share price growth is typically highly correlated with earnings growth. A strong or improving Net Profit Margin can signal a good investment opportunity, but it’s essential to analyze sustainability, industry context, and other financial indicators before making a decision."
            if value_24 > value_23:
                trend_rec = (
                        "An increased Net Profit Margin means a company is better at controlling costs and expenses relative to its revenue.")  
            else:
                trend_rec = (
                            "A decreased Net Profit Margin means a company is less effective at converting revenue into profit.")  
            if value_24 > industry_value:
                industry_rec = ("The Net Profit Margin of certain firm is higher than the industrial average means the firm is more effective at turning revenue into profit compared to its peers, indicating a efficient cost management, favorable pricing, or lower tax and interest expenses.")  
            
            else: 
                industry_rec = ("The Net Profit Margin of certain firm is lower than the industrial average means the firm is less efficient at converting revenue into profit relative to competitors, indicating a higher operating costs, weaker pricing power, or higher tax and interest burdens.")
            return f"{trend_rec} + {industry_rec} + {Investment_Recommendations}"
        
        elif metric == 'Diluted Normalised EPS':
            Investment_Recommendations = "Investors can use ROE to estimate a stock’s growth rate and the growth rate of its dividends by comparing with the industrial ROE. A larger ROE indicates a better profitability and may suggest an excepted corporate growth. However, an extremely high ROE can also be the result of a small equity account compared to net income, which indicates risk. Publicly listed companies with a long-term ROE of 10% to 20% are considered excellent companies. However, an ROE below 10% does not necessarily indicate poor company performance, depending on the stage of development of the company."
            if value_24 > value_23:
                trend_rec = (
                        "An increase in EPS means the firm is profitable enough to pay out more money to its shareholders. ")  
            else:
                trend_rec = (
                            "An decrease in EPS means that the firm is less profitable during this period, therefore, it has to cut down the money paid out to its shareholders. ")  
            if value_24 > industry_value:
                industry_rec = ("The EPS of the firm is higher than the industry average value, which means that it is performing relatively better than other players in the industry.")  
            
            else: 
                industry_rec = ("The EPS of the firm is lower than the industry average value, which means that it is not performing as well compared to other players in the industry.")
            return f"{trend_rec} + {industry_rec} + {Investment_Recommendations}"
       
        elif metric == 'Current Ratio':
            Investment_Recommendations = "Current ratios measure a company’s short-term liquidity, or its ability to generate enough cash to meet its short-term obligations. However, this ratio does not always necessarily give an accurate representation of liquidity because it includes inventory and other current assets that are more difficult to liquidate. Due to the nature of generalization, the current ratio tends to overstate a company’s liquidity. Thus, it must be taken with a grain of salt and definitely only used in conjunction with many other ratios and analysis."
            if value_24 > value_23:
                trend_rec = (
                        "The current ratio of the company increased last year, which indicates that the company is more liquid and has better coverage of outstanding debts")  
            else:
                trend_rec = (
                            "The current ratio of the company decreased last year, which shows the company has less short-term liquidity. ")  
            if value_24 > industry_value:
                industry_rec = ("This generally indicates stronger liquidity, meaning the company has more assets available to meet short-term obligations than its peers. A high current ratio can signal conservative management of assets or a strong cash position, which could make the company more resilient in downturns. However, if it’s too high, it might suggest inefficient use of assets; the company might be holding too much cash or inventory that could otherwise be invested for growth.")  
            
            else: 
                industry_rec = ("This may signal weaker liquidity, indicating the company has fewer assets to cover its short-term liabilities compared to peers. A low current ratio might indicate efficient use of resources if it has a strong cash flow or well-managed debt but could also pose a risk during financial distress if cash inflows are irregular. Investors often look for other financial indicators to understand if the low ratio is due to effective asset utilization or poor liquidity.")
            return f"{trend_rec} + {industry_rec} + {Investment_Recommendations}"
        
        elif metric == 'Quick Ratio':
            Investment_Recommendations = "The quick ratio, often referred to as the acid-test ratio, is another measure of a company’s financial health other than current ratio, which gauges short-term liquidity more rigorously by including only assets that can be converted to cash within 90 days or less. Companies may strive to keep its quick ratio between 0.1 and 0.25, though a quick ratio that is too high means a company may be inefficiently holding too much cash. Same as current ratio, quick ratio is most useful when it is used in comparative form, internally or externally. "
            if value_24 > value_23:
                trend_rec = (
                        "This suggests an improvement in liquidity, meaning the company is better positioned to meet its short-term obligations without relying on inventory. A higher quick ratio can indicate strong cash reserves or easily liquidated assets, which could improve the company's resilience in times of financial stress. However, if the ratio is increasing too much, it might also indicate underutilized assets that could otherwise be invested for growth.")  
            else:
                trend_rec = (
                            "This implies lower liquidity, meaning the company might face more difficulty covering its short-term liabilities with its most liquid assets. A declining quick ratio can be concerning if it falls below 1, as this suggests that liquid assets alone aren’t sufficient to cover immediate obligations. In some cases, a lower quick ratio might not necessarily be negative, especially if the company is investing cash into growth opportunities, though it may increase reliance on inventory or cash flow to manage liabilities.")  
            if value_24 > industry_value:
                industry_rec = ("recommendation3")  
            
            else: 
                industry_rec = ("recommendation4")
            return f"{trend_rec} + {industry_rec} + {Investment_Recommendations}"
        
        elif metric == 'Asset Turnover':
            Investment_Recommendations = "The asset turnover is useful as an indicator of whether companies are collecting receivables and turning inventory efficiently, in short, whether or not they are managing assets efficiently. However, it can be somewhat misleading since companies with large amounts of cash and short-term investments, which generally is a positive, may show a much lower turnover than companies with weak cash positions. Moreover, asset turnover ratios vary across different industry sectors, so there’s no specific range of good asset turnover ratio that applies to every company. The ratio needs to be used in comparative form by looking at the change over financial periods internally and comparing it with the value of other firms in the same industry."
            if value_24 > value_23:
                trend_rec = (
                        "An increase in asset turnover indicates that the company is collecting receivables and turning inventory efficiently.")  
            else:
                trend_rec = (
                            "An increase in asset turnover indicates that the company is not collecting receivables and turning inventory efficiently.")  
            if value_24 > industry_value:
                industry_rec = ("A higher asset turnover ratio suggests the company is efficiently using its assets to generate revenue compared to peers. This is often seen as a positive indicator of operational efficiency. High asset turnover can mean that the company has well-managed assets, effectively balancing its investments in assets with its revenue generation. In some cases, a high ratio could indicate that the company operates on low-margin, high-volume sales (common in retail or fast-moving consumer goods) and compensates for lower margins by selling quickly.")  
            
            else: 
                industry_rec = ("A lower asset turnover ratio indicates the company is less efficient in utilizing its assets to generate revenue compared to industry peers. This might suggest that the company has underutilized or idle assets, potentially pointing to inefficiencies in operations, overinvestment, or outdated assets that aren’t generating expected returns. In some industries, however, a lower asset turnover can be typical for companies that require significant asset investments (like machinery in manufacturing or infrastructure in utilities), so it may not necessarily indicate poor performance.")
            return f"{trend_rec} + {industry_rec} + {Investment_Recommendations}"
        return ""
    
    # 存储结果
    results = {}
    metrics_list = ['Asset Turnover', 'Current Ratio', 'Diluted Normalised EPS', 
                    'Net Profit Margin', 'Operating Margin', 'Quick Ratio', 
                    'Return on Assets (ROA)', 'Return on Equity (ROE)']
    for metric in metrics_list: 
        value_23 = year_metrics.loc[metric, 2023]
        value_24 = year_metrics.loc[metric, 2024]
        industry_value = industry_metrics[metric]
        # results.append(generate_result(metric, value_23, value_24, industry_value))
        # analysis_result = generate_result(metric, value_23, value_24, industry_value)
        # recommendation = generate_recommendation(metric, value_23, value_24, industry_value)
        
        analysis_result = generate_result(metric, value_23, value_24, industry_value)
        recommendation = generate_recommendation(metric, value_23, value_24, industry_value)
        
        results[metric] = {
            'Analysis': analysis_result,
            'Recommendation': recommendation
        }

    return results



def regulator_analyze_financial_metrics(company_data, industry_data):
   
    # 清理数据，确保列名一致，年份为数值类型
    company_data.columns = company_data.columns.str.strip()
    company_data['Year'] = pd.to_numeric(company_data['Year'], errors='coerce')

    # 筛选2023和2024年的数据
    metrics_comparison = company_data[company_data['Year'].isin([2023, 2024])][
        ['Year', 'Return on Equity (ROE)', 'Return on Assets (ROA)', 'Operating Margin', 'Net Profit Margin', 
         'Diluted Normalised EPS', 'Current Ratio', 'Asset Turnover', 'Quick Ratio']
    ]

    # 提取2024年的行业平均值
    industry_metrics = company_data[company_data['Year'] == 2024].iloc[0]

    # 提取2023年和2024年的指标数据
    year_metrics = metrics_comparison.set_index('Year').T

    def generate_result(metric, value_23, value_24, industry_value):
        trend_result = f"The company's {metric} is {'increasing' if value_24 > value_23 else 'decreasing'}, indicating "
        
        industry_comparison = f"The company's {metric} is {'above' if value_24 > industry_value else 'below'} the industry average."
        return f"{trend_result}\n{industry_comparison}"

    def generate_recommendation(metric, value_23, value_24, industry_value):
        if metric == 'Return on Assets (ROA)':
            if value_24 > value_23:
                trend_rec = (
                        "When ROA increases, regulators should evaluate whether the increase in ROA is driven by "
                        "improved asset utilization or one-time gains. If it's due to operational efficiency, it reflects "
                        "positively on the company's management. Regulators also should ensure the company isn't "
                        "compromising long-term stability for short-term gains." 
                )
            else:
                trend_rec = (
                            "When ROE decreases, regulators should assess whether the dropis temporary or indicative of"
                             "deeper structural issues. Regulators could encourage companies to optimize resource "
                             "allocation, improve cost management, or innovate to enhance profitability without "
                             "compromising financial stability."                
               )
            return f"{trend_rec}"
        
        elif metric == 'Return on Equity (ROE)':
            if value_24 > value_23:
                trend_rec = (
                        "When ROE increases, regulators should examine whether the increase in ROE is "
                        "driven by sustainable business operations or short-term financial maneuvers. "
                        "An ROE increase due to high leverage or risky investments could expose the company to financial instability."
                        "Regulators should encourage firms to maintain a healthy balance between profitability and risk" 
                )
            else:
                trend_rec = ("When ROE decreases, regulators should assess whether the dropis temporary or indicative of"
                             "deeper structural issues. Regulators could encourage companies to optimize resource "
                             "allocation, improve cost management, or innovate to enhance profitability without "
                             "compromising financial stability."                
                )
            return f"{trend_rec}"
        elif metric == 'Operating Margin':
            if value_24 > value_23:
                trend_rec = (
                        "When Operating Profit Margin increases, regulators should assess whether the increase in "
                        "margin is driven by sustainable factors such as improved operational efficiency or cost "
                        "control, rather than short-term factors like price hikes. If the increase stems from anti-competitive "
                        "behavior (e.g., monopolistic pricing or market dominance), regulators should investigate and ensure fair competition is maintained." 
                )
            else:
                trend_rec = ("When Operating Profit Margin decreases, regulators should encourage companies to optimize "
                             "operations and improve cost structures. If the decrease is due to external factors such as  "
                             "not engaging in harmful cost-cutting measures (e.g., underpaying workers or compromising "
                             "product quality)."                
                )
            return f"{trend_rec}"
        elif metric == 'Net Profit Margin':
            if value_24 > value_23:
                trend_rec = (
                        "When Net Profit Margin increases, regulators should evaluate whether the increase in net "
                        "profit margin is due to sustainable factors like operational improvements or tax efficiency, "
                        "rather than one-time events such as asset sales or subsidies. If the increase results from "
                        "pricing strategies or cost-cutting measures that might harm consumers (e.g., price gouging or " 
                        "reducing product quality), regulators should intervene to ensure market fairness."
                )
            else:
                trend_rec = ("When Net Profit Margin decreases, regulators should assess whether these factors pose a "
                             "systemic risk or reflect temporary challenges. Regulators may advise firms to strengthen their "
                             "financial health through cost optimization and diversification, avoiding aggressive cost-cutting "
                             "that could impact stakeholders."                
                )
            return f"{trend_rec}"
        elif metric == 'Diluted Normalised EPS':
            if value_24 > value_23:
                trend_rec = (
                        "When a firm’s EPS increases, regulators might recommend that companies consider reinvesting "
                        "these increased earnings into growth initiatives, distributing dividends, buying back shares, or "
                        "improving employee benefits, to ensure long-term shareholder value and company stability. It’s "
                        "important for companies to be transparent about the sources of earnings growth to build investor " 
                        "confidence and avoid any perception of financial manipulation. Regulators would also "
                        "discourage firms from artificially inflating EPS through measures like excessive stock "
                        "buybacks, which can make EPS appear stronger without any underlying improvement in operations."
                )
            else:
                trend_rec = ("When a firm’s EPS decreases, regulators might encourage companies to provide a clear "
                             "disclosure of the causes behind this decline, particularly if it reflects financial or operational "
                             "difficulties. Offering forward-looking guidance can reassure investors, especially if the EPS "
                             "drop is part of a broader industry trend, while regulators may also caution firms against taking "
                             "on excessive risk in a bid to boost short-term earnings."                
                )
            return f"{trend_rec}"
        elif metric == 'Current Ratio':
            if value_24 > value_23:
                trend_rec = (
                        "An increase in the Current Ratio signals improved liquidity, and regulators may recommend "
                        "that companies leverage this liquidity effectively, perhaps through strategic investments or "
                        "addressing outstanding liabilities. It is advisable that companies avoid hoarding liquid assets "
                        "without a clear purpose, as this may suggest inefficiencies or lack of a proactive growth " 
                        "strategy. Increased transparency in financial reporting is essential here, especially to clarify "
                        "whether the higher ratio results from stronger cash flows or reduced liabilities. "
                )
            else:
                trend_rec = ("When Current Ratio decreases, regulators might suggest that firms focus on maintaining "
                             "sufficient liquidity to guard against short-term financial challenges, especially during "
                             "uncertain economic conditions. Refinancing strategies and liability management can be "
                             "beneficial to ensure the firm can meet its obligations. It’s also important for firms to clearly " 
                             "communicate the reasons for reduced liquidity to their investors, particularly if this reduction "    
                             "is driven by rising debt or other operational challenges."           
                )
            return f"{trend_rec}"
        elif metric == 'Quick Ratio':
            if value_24 > value_23:
                trend_rec = (
                        "With a rising Quick Ratio, regulators may advise companies to ensure their additional cash or "
                        "liquid assets are used efficiently and not left idle. It is beneficial for firms to provide clarity "
                        "on the reasons behind this increase, whether it stems from stronger cash flows, the sale of "
                        "assets, or other sources, to avoid misleading stakeholders. Balancing liquidity with growth " 
                        "investments is key to preventing an overly conservative approach that could limit potential gains. "
                )
            else:
                trend_rec = ("When Quick Ratio decreases, regulators regulators might recommend that companies take steps to improve "
                             "liquidity management, especially to avoid future solvency risks. Increased transparency "
                             "around declining liquidity is crucial, as it enables investors to understand whether it’s due to "
                             "issues like poor receivables collections or rising liabilities. Regulators may also advise "     
                             "companies to avoid taking on extensive risks, such as significant credit extensions or "    
                             "aggressive expansion, when liquidity is already under strain."       
                )
            return f"{trend_rec}"
        elif metric == 'Asset Turnover':
            if value_24 > value_23:
                trend_rec = (
                        "When a company’s Asset Turnover ratio increases, regulators may encourage firms to continue "
                        "their focus on efficient asset utilization but to do so in a way that doesn’t exhaust resources "
                        "unsustainably. Disclosing the factors behind this efficiency—whether through process "
                        "improvements, divesting non-essential assets, or other measures—can give investors better " 
                        "insight into the nature of the company’s efficiency gains. Ensuring that assets are used "
                        "sustainably is especially important in industries that require ongoing reinvestment. "
                )
            else:
                trend_rec = ("When a company’s Asset Turnover ratio decreases, regulators may recommend that firms address inefficiencies in "
                             "their asset management, such as restructuring or repurposing underperforming assets to "
                             "increase their contribution to revenue. Transparency with investors regarding the causes of a "
                             "declining asset turnover, such as slower sales or heavy investments in growth areas, helps to " 
                             "maintain investor trust. Regulators might also advise firms to consider divesting or finding " 
                             "new uses for idle assets if these are unlikely to generate satisfactory returns in the near future, "    
                             "improving overall operational efficiency."          
                )
            return f"{trend_rec}"
        return ""

    # 存储结果
    results = {}
    metrics_list = ['Asset Turnover', 'Current Ratio', 'Diluted Normalised EPS', 
                    'Net Profit Margin', 'Operating Margin', 'Quick Ratio', 
                    'Return on Assets (ROA)', 'Return on Equity (ROE)']
    for metric in metrics_list:
        value_23 = year_metrics.loc[metric, 2023]
        value_24 = year_metrics.loc[metric, 2024]
        industry_value = industry_metrics[metric]
        
        analysis_result = generate_result(metric, value_23, value_24, industry_value)
        recommendation = generate_recommendation(metric, value_23, value_24, industry_value)
        
        results[metric] = {
            'Analysis': analysis_result,
            'Recommendation': recommendation
        }
    
    return results