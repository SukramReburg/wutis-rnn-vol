from indicators import *
import yaml
import os

if __name__ == "__main__":
    indicators_impl = {'rsi': RSI, 
                  'ma': MovingAverage, 
                  'ema': Exponential, 
                  'bb': BollingerBands, 
                  'macd': MACD, 
                  'atr': ATR, 
                  'stoch': StochasticOscillator, 
                  'adx': ADX, 
                  'wr': WilliamsR, 
                  'cci': CCI, 
                  'obv': OBV, 
                  'mfi': MFI, 
                  'cmo': CMO, 
                  'aroon': AROON
                  }
    with open('config/data_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    indicators = config['indicators']
    tickers = config['indicators_for_tickers']
    paths = config['paths']

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, paths['processed'])
    os.makedirs(path, exist_ok=True)


    for ticker in tickers:
        print(f"Processing data for {ticker}...")
        path = os.path.join(base_dir, paths['raw'], f"{ticker}.csv")
        data = pd.read_csv(path)

        for indicator_def in indicators:
            name = indicator_def['name']
            params = indicator_def.get('params', {})

            if name in indicators_impl:
                cls = indicators_impl[name]
                instance = cls(data, **params) 
                instance.add_indicator()
                data = instance.data
                print(f"Processed: {name.upper()}")
            else:
                print(f"Unknown indicator: {name}")
        data.to_csv(os.path.join(base_dir, paths['processed'], f"{ticker}.csv"), index=False)
        