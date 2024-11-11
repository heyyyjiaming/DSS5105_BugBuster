import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from yahooquery import search
import pandas as pd
import datetime
from bs4 import BeautifulSoup
import json
import requests
import re
from io import StringIO, BytesIO


# def get_ticker_symbol(company_name):
#     # Use yahooquery's search function to get the ticker symbol
#     result = search(company_name)
#     print(result)
#     # if 'quotes' in result and len(result['quotes']) > 0:
#     #     # Return the first matching ticker symbol
#     #     ticker_symbol = result['quotes'][0]['symbol']
#     #     return ticker_symbol
#     # else:
#     #     return None
# def get_ticker_symbol(company_name):
#     try:
#         # Perform search using yahooquery
#         result = search(company_name)
#         print(f"Raw response: {result}")  # Debug print to see response

#         # Check if response is in the expected format
#         if not isinstance(result, dict) or 'quotes' not in result:
#             print("Unexpected response format or empty response")
#             return None
        
#         # Extract the ticker symbol from the result
#         quotes = result.get('quotes', [])
#         if quotes:
#             return quotes[0].get('symbol', None)
#         else:
#             print("No quotes found for the company name")
#             return None

#     except requests.exceptions.JSONDecodeError as e:
#         print("JSONDecodeError: Unable to parse JSON response")
#         return None
#     except requests.exceptions.RequestException as e:
#         print(f"Request Error: {e}")
#         return None

# def get_stock_data(company_name):
#     # Map company name to ticker symbol
#     ticker_symbol = get_ticker_symbol(company_name)
#     if not ticker_symbol:
#         print(f"Can't get ticker symbol for {company_name}")
#         return

#     # Fetch data from 2020-01-01 to today
#     start_date = '2020-01-01'
#     end_date = datetime.datetime.today().strftime('%Y-%m-%d')
#     data = yf.download(ticker_symbol, start=start_date, end=end_date)
    
#     # Check if data is available
#     if data.empty:
#         print(f"No data available for {company_name} ({ticker_symbol})")
#         return

#     # Check if 'Adj Close' column exists; if not, use 'Close' as a fallback
#     if 'Adj Close' in data.columns:
#         data = data[['Adj Close']]
#     elif 'Close' in data.columns:
#         data = data[['Close']]
#         data.rename(columns={'Close': 'Adj Close'}, inplace=True)
#     else:
#         print(f"No 'Adj Close' or 'Close' data available for {company_name} ({ticker_symbol})")
#         return
    
#     # Format date and reset index
#     data.reset_index(inplace=True)
#     data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
#     data.columns = data.columns.droplevel(1)

    
#     return data



def clean_name(name):
    # Remove all Chinese characters and special symbols, retaining only letters, numbers, and spaces
    name = re.sub(r'[^\x00-\x7F]+', '', name)  # Remove non-ASCII characters
    name = re.sub(r'[^\w\s]', '', name)  # Remove special symbols except alphanumeric and whitespace
    name = re.sub(' ', '', name)  # Remove spaces 
    name = name.lower() # Convert all uppercase letters to lowercase
    return name.strip()

# mydata = pd.read_csv('mydata.csv')
# # Step 1: Rename column titles to English
# mydata.columns = ["Company_Name", "Transaction_Code", "RIC", "Market_Value_Million", 
#               "Total_Revenue_Million", "P_E_Ratio", "Yield_Percent", "Sector", "GTI_Score"]

# # Step 2: Clean the 'Company_Name' column by:
# mydata['cleaned_transaction_name'] = mydata['Company_Name'].apply(clean_name)
# new_data = mydata[['cleaned_transaction_name','Transaction_Code','RIC']
#                   ]
# # Extract the part of 'RIC' after the period and create the 'company_ticker' column
# mydata['company_ticker'] = mydata['Transaction_Code'] + '.' + mydata['RIC'].str.split('.').str[-1]

# # Display the updated dataframe
# new_data = mydata[['cleaned_transaction_name', 'Transaction_Code', 'RIC', 'company_ticker']]

# new_data.to_csv('company_ticker_mapping.csv')


def get_company_ticker(company_name, company_ticker_map):
    # company_ticker_map = pd.read_csv('company_ticker_mapping.csv')
    # company_url = "https://raw.githubusercontent.com/heyyyjiaming/DSS5105_BugBuster/refs/heads/main/tests/FinancialData/company_ticker_mapping.csv"
    # response = requests.get(company_url)
    
    # if response.status_code == 200:
    #     company_ticker_map = pd.read_excel(BytesIO(response.content), header=0, engine='openpyxl')
    # else:
    #     st.error("Failed to load data from GitHub.")
    
    
    # Find the row in new_data that matches the company_name
    cleaned_input_name= clean_name(company_name)
    result = company_ticker_map[company_ticker_map['cleaned_transaction_name'] == cleaned_input_name]
    
    # If a matching row is found, return the 'company_ticker' value
    if not result.empty:
        ticker = result['company_ticker']
        return result.iloc[0]['company_ticker']
    else:
        return None  # Return None if no match is found


def get_stock_data(company_name, company_ticker_map):
    # Map company name to ticker symbol
    ticker_symbol = get_company_ticker(company_name, company_ticker_map)
    if not ticker_symbol:
        print(f"Can't get ticker symbol for {company_name}")
        return

    # Fetch data from 2020-01-01 to today
    start_date = '2020-01-01'
    # end_date = datetime.datetime.today().strftime('%Y-%m-%d')
    end_date = datetime.date.today().strftime('%Y-%m-%d')
    data = yf.download(ticker_symbol, start=start_date, end=end_date)
    
    # Check if data is available
    if data.empty:
        print(f"No data available for {company_name} ({ticker_symbol})")
        return

    # Check if 'Adj Close' column exists; if not, use 'Close' as a fallback
    if 'Adj Close' in data.columns:
        data = data[['Adj Close']]
    elif 'Close' in data.columns:
        data = data[['Close']]
        data.rename(columns={'Close': 'Adj Close'}, inplace=True)
    else:
        print(f"No 'Adj Close' or 'Close' data available for {company_name} ({ticker_symbol})")
        return
    
    # Format date and reset index
    data.reset_index(inplace=True)
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')

    # Create folder if it doesn't exist
    folder_path = 'StockPrice'
    os.makedirs(folder_path, exist_ok=True)

    # Save data as CSV with company name in filename
    csv_filename = f"{folder_path}/{company_name}_stock_data.csv"
    data.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}")

    # Retrieve the most recent adjusted close price if real-time data is unavailable
    ticker = yf.Ticker(ticker_symbol)
    history = ticker.history(period='1d')
    if 'Adj Close' in history.columns:
        current_price = history['Adj Close'].iloc[-1]
    elif 'Close' in history.columns:
        current_price = history['Close'].iloc[-1]
    else:
        current_price = None

    if current_price is not None:
        print(f"{company_name} ({ticker_symbol}) most recent Adj Close: ${current_price}")
    else:
        print(f"No recent 'Adj Close' or 'Close' price available for {company_name} ({ticker_symbol})")






def get_esg_news(company_name, api_key):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
    "q": f"{company_name} Inc, ESG",
    "tbs": "qdr:m"
    })
    headers = {
    'X-API-KEY': api_key,
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    result = pd.json_normalize(response.json()['organic']).iloc[:, 0:-2]
    
    return result

