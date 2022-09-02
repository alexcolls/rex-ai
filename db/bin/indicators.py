import pandas as pd
import numpy as np
import math
import scipy.signal as sig


def time_standard(df:pd.DataFrame):
    # Splitting Datetime
    df.index = pd.to_datetime(df.index)
    df["month_sin"] = np.sin((np.array(df.index.month)*math.pi*2)/12)
    df["month_cos"]=np.cos((np.array(df.index.month)*math.pi*2)/12)
    df["day_sin"] = np.sin((np.array(df.index.day_of_year)*math.pi*2)/360)
    df["day_cos"]=np.cos((np.array(df.index.day_of_year)*math.pi*2)/360)
    df["weekday_sin"] = np.sin((np.array(df.index.weekday)*math.pi*2)/7)
    df["weekday_cos"]=np.cos((np.array(df.index.weekday)*math.pi*2)/7)
    df["hour_sin"] = np.sin((np.array(df.index.hour)*math.pi*2)/24)
    df["hour_cos"]=np.cos((np.array(df.index.hour)*math.pi*2)/24)

    return df



# VOLATILITY INDICATORS

def get_bollinger_bands(df, rate=5, sigma_start=2, sigma_stop=2):
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


def bollinger_small(df, rate=24, sigma_start=2, sigma_stop=2):

    data = pd.DataFrame([])
    for currency in df.columns:
        data[f'{currency}_sma'] = df[currency].rolling(rate).mean()
        data[f'{currency}_std']= df[currency].rolling(rate).std()
    data.index = df.index

    return data

def volatility(df,rate=240,window=506):

    data = pd.DataFrame([])
    for currency in df.columns:
        data[f"{currency}_vol1"] = df[currency].rolling(rate).std()/(window**0.5)
        data[f"{currency}_vol2"] = df[currency].rolling(rate).std()*(window**0.5)
    data.index = df.index

    return data




def atr(idxs: pd.DataFrame, low: pd.DataFrame, high: pd.DataFrame, window:int=14):
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



# TENDENCY INDICATORS

def ema(df, window=48):
    """
    Exponential movint average
    """
    data = pd.DataFrame([])
    for currency in df.columns:
        data[f'{currency}_ema'] = df[currency].ewm(alpha=1/window, adjust=False).mean()
    data.index = df.index

    return data


def highpass_filter(df, order=5, cutoff=0.2):
    """Highpass filter """

    data = pd.DataFrame([])
    b, a = sig.butter(N=order, Wn=cutoff, btype='highpass', analog=False)
    for currency in df.columns:
        data[f'{currency}_high'] = sig.lfilter(b, a, df[currency])
    data.index = df.index

    return data

def lowpass_filter(df, order=8, cutoff=0.2):
    """Lowpass filter """
    data = pd.DataFrame([])
    b, a = sig.butter(N=order, Wn=cutoff, btype='lowpass', analog=False)
    for currency in df.columns:
        data[f'{currency}_low'] = sig.lfilter(b, a, df[currency])
    data.index = df.index

    return data

def rsi(df, periods = 240):
    """Returns a series with the rsi """
    data = pd.DataFrame([])
    for currency in df.columns:
        delta = df[currency].diff()
        up = delta.clip(lower=0)
        down = -1*delta.clip(upper=0)
        ema_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
        ema_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
        rs = ema_up / ema_down
        rsi = 100 - (100/(1 + rs))
        data[f'{currency}_rsi'] = rsi

    return data