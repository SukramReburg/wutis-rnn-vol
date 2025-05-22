import yaml
import pandas as pd
import os  

from alpaca.data.historical.stock import StockBarsRequest,StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame

from datetime import datetime

def get_prev_data(ticker,start_year):

    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    api_key = config['alpaca_api']['api_key']
    secret_key = config['alpaca_api']['secret_key']

    start_dt = datetime(start_year,1,1,0,0,0,0) # TODO: bettter date handling, more options for prev data sources
    request_params = StockBarsRequest(          # implement in a class?
                        symbol_or_symbols=ticker,
                        timeframe=TimeFrame.Minute,
                        start=start_dt,
                        end=datetime.now().replace(day=datetime.now().day -1)
                )
    client = StockHistoricalDataClient(api_key,secret_key)
    return client.get_stock_bars(request_params).df

if __name__ == "__main__":

    with open('config/data_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    tickers = config['tickers']
    start_year = config['start_year']
    
    paths = config['paths']
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, paths['raw'])
    os.makedirs(path, exist_ok=True)

    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        df = get_prev_data(ticker,start_year)
        df.to_csv(os.path.join(path, f"{ticker}.csv"))