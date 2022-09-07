
# author: Marti Llanes
# license: MIT

import pandas as pd
import numpy as np
import math
import scipy.signal as sig
from config import SYMBOLS


### INDICATORS LIBRARY ###

def time_standard( self, df ):
    # spliting datetime
    data = pd.DataFrame([])
    data.index = pd.to_datetime(df.index) # type: ignore
    data["month_sin"] = np.sin((np.array(data.index.month)*math.pi*2)/12)
    data["month_cos"]=np.cos((np.array(data.index.month)*math.pi*2)/12)
    data["day_sin"] = np.sin((np.array(data.index.day_of_year)*math.pi*2)/360)
    data["day_cos"]=np.cos((np.array(data.index.day_of_year)*math.pi*2)/360)
    data["weekday_sin"] = np.sin((np.array(data.index.weekday)*math.pi*2)/7)
    data["weekday_cos"]=np.cos((np.array(data.index.weekday)*math.pi*2)/7)
    data["hour_sin"] = np.sin((np.array(data.index.hour)*math.pi*2)/24)
    data["hour_cos"]=np.cos((np.array(data.index.hour)*math.pi*2)/24)
    data.index = df.index

    return data

def correlations( self, df, window=120 ):

    data = pd.DataFrame([])
    for sym in SYMBOLS:
        data[f'{sym[:3]}_{sym[4:]}_corr'] = df[f'{sym[:3]}'].rolling(rate).corr(df[f'{sym[4:]}'])
    data.index = df.index
    return data

def get_bollinger_bands( self, df, rate=5, sigma_start=2, sigma_stop=2):
    """regular bollinger bands, do not apply to a RNN, use bollinger_small"""
    sma = df.rolling(rate).mean()
    bollinger_bands = {"sma":sma}
    std = df.rolling(rate).std()
    for sigma in range(sigma_start, sigma_stop+1):
        up = sma + std * sigma
        down = sma - std * sigma
        bollinger_bands[f'up{sigma}'] = up
        bollinger_bands[f'down{sigma}'] = down
    bol = pd.DataFrame.from_dict(bollinger_bands)
    bol.set_index(df.index, inplace=True)
    return bol

def bollinger_small( self, df, rate=24, n_devs=[ 1, 2, 3 ]):

    data = pd.DataFrame([])
    for currency in df.columns:
        sma = df[currency].rolling(rate).mean()
        std  = df[currency].rolling(rate).std()
        data[f'{currency}_sma'] = sma
        data[f'{currency}_std'] = std
        for i in n_devs:
            data[f'{currency}_+{i}std'] = sma + std * i
            data[f'{currency}_-{i}std'] = sma - std * i
    data.index = df.index

    return data

def volatility( self, df, rate=240, window=506 ):

    data = pd.DataFrame([])
    for currency in df.columns:
        data[f"{currency}_vol1"] = df[currency].rolling(rate).std()/(window**0.5)
        data[f"{currency}_vol2"] = df[currency].rolling(rate).std()*(window**0.5)
    data.index = df.index

    return data

def sharpe_ratio( self, df, window=24 ):

    data = pd.DataFrame([])
    for currency in df.columns:
        data[f'{currency}_sharpe'] = df[currency].rolling(window).mean()/df[currency].rolling(window).std()
    data.index = df.index

    return data

def atr( self, idxs: pd.DataFrame, low: pd.DataFrame, high: pd.DataFrame, window:int=14):

    data = pd.DataFrame([])

    for currency in idxs.columns:
        data[f'idxs_{currency}'] = idxs[currency]
        data[f'high_{currency}'] = high[currency]
        data[f'low_{currency}'] = low[currency]
        data[f'tr0_{currency}'] = data[f'high_{currency}'] - data[f'low_{currency}']
        data[f'tr1_{currency}'] = np.abs(data[f'high_{currency}'] - data[f'idxs_{currency}'].shift())
        data[f'tr2_{currency}'] = np.abs(data[f'low_{currency}'] - data[f'idxs_{currency}'].shift())
        data[f'tr_{currency}'] = data[[f'tr0_{currency}', f'tr1_{currency}', f'tr2_{currency}']].max(axis=1)
        data[f'atr_{currency}'] = data[f'tr_{currency}'].ewm(alpha=1/window, adjust=False).mean()
        data.drop(columns=[f'idxs_{currency}',f'tr0_{currency}',f'tr1_{currency}',f'tr2_{currency}',f'tr_{currency}'], inplace=True)
    
    return data

def ema( self, df, window=48 ):
    """
    Exponential moving average
    """
    data = pd.DataFrame([])
    for currency in df.columns:
        data[f'{currency}_ema{window}'] = df[currency].ewm(alpha=1/window, adjust=False).mean()
    data.index = df.index

    return data

def highpass_filter( self, df, order=5, cutoff=0.2):

    data = pd.DataFrame([])
    b, a = sig.butter(N=order, Wn=cutoff, btype='highpass', analog=False)
    for currency in df.columns:
        data[f'{currency}_highpass'] = sig.lfilter(b, a, df[currency])
    data.index = df.index

    return data

def lowpass_filter( self, df, order=8, cutoff=0.2 ):
    
    data = pd.DataFrame([])
    b, a = sig.butter(N=order, Wn=cutoff, btype='lowpass', analog=False)
    for currency in df.columns:
        data[f'{currency}_lowpass'] = sig.lfilter(b, a, df[currency])
    data.index = df.index

    return data

def rsi( self, df, window=240 ):
    
    data = pd.DataFrame([])
    for currency in df.columns:
        delta = df[currency].diff()
        up = delta.clip(lower=0)
        down = -1*delta.clip(upper=0)
        ema_up = up.ewm(com = window - 1, adjust=True, min_periods=window).mean()
        ema_down = down.ewm(com=window - 1, adjust=True, min_periods=window).mean()
        rs = ema_up / ema_down
        rsi = 100 - (100/(1 + rs))
        data[f'{currency}_rsi{window}'] = rsi

    return data

def lowpass_momentum( self, df ):

    data = pd.DataFrame([])
    low = self.lowpass_filter(df)
    low2 = low.shift(1)
    for currency in low.columns:
        data[f'{currency}_hp_momentum'] = low2[currency] - low[currency]
    data.index = df.index
    data.fillna(0)
    
    return data

def highpass_momentum( self, df ):

    data = pd.DataFrame([])
    high = self.highpass_filter(df)
    high2 = high.shift(1)
    for currency in high.columns:
        data[f'{currency}_hp_momentum'] = high2[currency] - high[currency]
    data.index = df.index
    data.fillna(0, inplace=True)
    
    return data

def ema_diff( self, df, window=8 ):

    data = pd.DataFrame([])
    for currency in df.columns:
        data[f'{currency}_ema{window}_diff'] = df[currency].ewm(alpha=1/window, adjust=False).mean() - df[currency]
    data.index = df.index
    data.fillna(0, inplace=True)

    return data
