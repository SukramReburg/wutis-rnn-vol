import pandas as pd

class Indicator(): 
    def __init__(self, data, name=None): 
        self.data = data 
        self.name = name

    def calculate(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def add_indicator(self):
        result = self.calculate()
        if isinstance(result, pd.DataFrame):
            for col in result.columns:
                self.data[col] = result[col]
        else:
            self.data[self.name] = result
        return self.data

class MovingAverage(Indicator): 
    def __init__(self, data, period, name='ma'):
        super().__init__(data, name)
        self.period = period

    def calculate(self):
        return self.data['close'].rolling(window=self.period).mean()
    
class Exponential(Indicator): 
    def __init__(self, data, period, name='ema'):
        super().__init__(data,name)
        self.period = period

    def calculate(self):
        return self.data['close'].ewm(span=self.period, adjust=False).mean()
    
class RSI(Indicator):
    def __init__(self, data, period, name='rsi'):
        super().__init__(data,name)
        self.period = period

    def calculate(self):
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

class ATR(Indicator):
    def __init__(self, data, period,name='atr'):
        super().__init__(data,name)
        self.period = period

    def calculate(self):
        high_low = self.data['high'] - self.data['low']
        high_close = abs(self.data['high'] - self.data['close'].shift())
        low_close = abs(self.data['low'] - self.data['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=self.period).mean()
        return atr

class WilliamsR(Indicator):
    def __init__(self, data, period,name='wr'):
        super().__init__(data,name)
        self.period = period

    def calculate(self):
        high_max = self.data['high'].rolling(window=self.period).max()
        low_min = self.data['low'].rolling(window=self.period).min()
        wr = -100 * ((high_max - self.data['close']) / (high_max - low_min))
        return wr
    
class CCI(Indicator):
    def __init__(self, data, period,name='cci'):
        super().__init__(data,name)
        self.period = period

    def calculate(self):
        typical_price = (self.data['high'] + self.data['low'] + self.data['close']) / 3
        sma = typical_price.rolling(window=self.period).mean()
        mad = (typical_price - sma).abs().rolling(window=self.period).mean()
        cci = (typical_price - sma) / (0.015 * mad)
        return cci

class ADX(Indicator):
    def __init__(self, data, period,name='adx'):
        super().__init__(data,name)
        self.period = period

    def calculate(self):
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']

        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        atr = tr.rolling(window=self.period).mean()

        plus_dm = high.diff().where((high.diff() > low.diff()) & (high.diff() > 0), 0)
        minus_dm = low.diff().where((low.diff() > high.diff()) & (low.diff() > 0), 0)

        plus_di = 100 * (plus_dm.rolling(window=self.period).sum() / atr)
        minus_di = 100 * (minus_dm.rolling(window=self.period).sum() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=self.period).mean()

        return adx

class OBV(Indicator):
    def __init__(self, data,name='obv'):
        super().__init__(data,name)

    def calculate(self):
        obv = (self.data['volume'] * (self.data['close'].diff() > 0).astype(int) - 
               self.data['volume'] * (self.data['close'].diff() < 0).astype(int)).cumsum()
        return obv

class MFI(Indicator):
    def __init__(self, data, period,name='mfi'):
        super().__init__(data,name)
        self.period = period

    def calculate(self):
        typical_price = (self.data['high'] + self.data['low'] + self.data['close']) / 3
        money_flow = typical_price * self.data['volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(window=self.period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(window=self.period).sum()
        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        return mfi

class CMO(Indicator):
    def __init__(self, data, period,name='cmo'):
        super().__init__(data,name)
        self.period = period

    def calculate(self):
        delta = self.data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).sum()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).sum()
        cmo = (gain - loss) / (gain + loss) * 100
        return cmo

class BollingerBands(Indicator):   
    def __init__(self, data, period, num_std_dev, name='bb'):
        super().__init__(data, name)
        self.period = period
        self.num_std_dev = num_std_dev

    def calculate(self):
        rolling_mean = self.data['close'].rolling(window=self.period).mean()
        rolling_std = self.data['close'].rolling(window=self.period).std()
        upper_band = rolling_mean + (rolling_std * self.num_std_dev)
        lower_band = rolling_mean - (rolling_std * self.num_std_dev)
        return pd.DataFrame({
            f'{self.name}_upper': upper_band,
            f'{self.name}_lower': lower_band
        }, index=self.data.index)


class MACD(Indicator):
    def __init__(self, data, short_window=12, long_window=26, signal_window=9, name='macd'):
        super().__init__(data, name)
        self.short_window = short_window
        self.long_window = long_window
        self.signal_window = signal_window

    def calculate(self):
        short_ema = self.data['close'].ewm(span=self.short_window, adjust=False).mean()
        long_ema = self.data['close'].ewm(span=self.long_window, adjust=False).mean()
        macd_line = short_ema - long_ema
        signal_line = macd_line.ewm(span=self.signal_window, adjust=False).mean()
        return pd.DataFrame({
            f'{self.name}_line': macd_line,
            f'{self.name}_signal': signal_line
        }, index=self.data.index)


class StochasticOscillator(Indicator):
    def __init__(self, data, period, smooth_k=3, smooth_d=3, name='stoch'):
        super().__init__(data, name)
        self.period = period
        self.smooth_k = smooth_k
        self.smooth_d = smooth_d

    def calculate(self):
        low_min = self.data['low'].rolling(window=self.period).min()
        high_max = self.data['high'].rolling(window=self.period).max()
        k = 100 * ((self.data['close'] - low_min) / (high_max - low_min))
        d = k.rolling(window=self.smooth_d).mean()
        k_smooth = k.rolling(window=self.smooth_k).mean()
        return pd.DataFrame({
            f'{self.name}_k': k_smooth,
            f'{self.name}_d': d
        }, index=self.data.index)


class AROON(Indicator):
    def __init__(self, data, period, name='aroon'):
        super().__init__(data, name)
        self.period = period

    def calculate(self):
        aroon_up = self.data['high'].rolling(window=self.period).apply(lambda x: (self.period - x[::-1].argmax()) / self.period * 100)
        aroon_down = self.data['low'].rolling(window=self.period).apply(lambda x: (self.period - x[::-1].argmin()) / self.period * 100)
        return pd.DataFrame({
            f'{self.name}_up': aroon_up,
            f'{self.name}_down': aroon_down
        }, index=self.data.index)