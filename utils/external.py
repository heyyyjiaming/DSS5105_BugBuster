import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from yahooquery import search
import pandas as pd
import datetime
from bs4 import BeautifulSoup
import json
import requests


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
def get_ticker_symbol(company_name):
    try:
        # Perform search using yahooquery
        result = search(company_name)
        print(f"Raw response: {result}")  # Debug print to see response

        # Check if response is in the expected format
        if not isinstance(result, dict) or 'quotes' not in result:
            print("Unexpected response format or empty response")
            return None
        
        # Extract the ticker symbol from the result
        quotes = result.get('quotes', [])
        if quotes:
            return quotes[0].get('symbol', None)
        else:
            print("No quotes found for the company name")
            return None

    except requests.exceptions.JSONDecodeError as e:
        print("JSONDecodeError: Unable to parse JSON response")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
        return None

def get_stock_data(company_name):
    # Map company name to ticker symbol
    ticker_symbol = get_ticker_symbol(company_name)
    if not ticker_symbol:
        print(f"Can't get ticker symbol for {company_name}")
        return

    # Fetch data from 2020-01-01 to today
    start_date = '2020-01-01'
    end_date = datetime.datetime.today().strftime('%Y-%m-%d')
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
    data.columns = data.columns.droplevel(1)

    
    return data


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

