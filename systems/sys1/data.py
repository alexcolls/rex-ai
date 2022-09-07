# author: Quantium Rock
# license: MIT

import os
import math
import numpy as np
import pandas as pd
import scipy.signal as sig
from pathlib import Path
from apis.oanda_api import OandaApi
from config import SYMBOLS, TIMEFRAME, LOOKBACK


class DataSet:

    def __init__(self, symbols=SYMBOLS, timeframe=TIMEFRAME, lookback=LOOKBACK):

        # quotes granularity
        self.symbols = symbols
        self.timeframe = timeframe
        self.lookback = lookback
        self.ccys = self.getCcys()
        self.db_path = os.path.normpath(
            os.path.join( os.path.dirname(os.path.abspath(__file__)), "../data/series/" )
        )
        # create directories if doesn't exist
        Path(self.db_path).mkdir(parents=True, exist_ok=True)

    # get data by year
    def getCandles( self ):

        oanda_api = OandaApi()

        # init daily dataframes indices for asks & bids
        op = pd.DataFrame()
        hi = pd.DataFrame()
        lo = pd.DataFrame()
        cl = pd.DataFrame()
        vo = pd.DataFrame()

        # iterate each instrument to get a full week data each
        for symbol in self.symbols:

            print(symbol)

            # initialize symbol data struct
            data = {
                "dtime": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
            }

            # request 5000 bars from oanda rest-api
            req = oanda_api.getLastCandles(symbol, self.timeframe, count=self.lookback)

            # iterate each candle
            for x in req:
                # append data
                data["dtime"].append(x["time"])
                data["open"].append(float(x["mid"]["o"]))
                data["high"].append(float(x["mid"]["h"]))
                data["low"].append(float(x["mid"]["l"]))
                data["close"].append(float(x["mid"]["c"]))
                data["volume"].append(int(x["volume"]))

            # ^ finished symbol year

            # transform data to prices dataframe
            _op = pd.DataFrame(data["open"], index=data["dtime"], columns=[symbol])
            _op.index = pd.to_datetime(_op.index, utc=True)
            _hi = pd.DataFrame(data["high"], index=data["dtime"], columns=[symbol])
            _hi.index = pd.to_datetime(_hi.index, utc=True)
            _lo = pd.DataFrame(data["low"], index=data["dtime"], columns=[symbol])
            _lo.index = pd.to_datetime(_lo.index, utc=True)
            _cl = pd.DataFrame(data["close"], index=data["dtime"], columns=[symbol])
            _cl.index = pd.to_datetime(_cl.index, utc=True)
            _vo = pd.DataFrame(data["volume"], index=data["dtime"], columns=[symbol])
            _vo.index = pd.to_datetime(_vo.index, utc=True)

            op = pd.merge(op, _op, how="outer", left_index=True, right_index=True)
            hi = pd.merge(hi, _hi, how="outer", left_index=True, right_index=True)
            lo = pd.merge(lo, _lo, how="outer", left_index=True, right_index=True)
            cl = pd.merge(cl, _cl, how="outer", left_index=True, right_index=True)
            vo = pd.merge(vo, _vo, how="outer", left_index=True, right_index=True)

            # realese memory
            del data, _op, _hi, _lo, _cl, _vo

        # ^ finished all symbols

        # fill nans with forward-fill (last non-nan price)
        op.fillna(method="ffill", inplace=True)
        hi.fillna(method="ffill", inplace=True)
        lo.fillna(method="ffill", inplace=True)
        cl.fillna(method="ffill", inplace=True)

        # fill nans with backward-fill (for the first seconds of the week)
        op.fillna(method="bfill", inplace=True)
        hi.fillna(method="bfill", inplace=True)
        lo.fillna(method="bfill", inplace=True)
        cl.fillna(method="bfill", inplace=True)

        # fill volume nans with 0
        vo.fillna(0, inplace=True)

        return op, hi, lo, cl, vo

    # get data by year
    def normalizeData( self, op, hi, lo, cl ):

        # create portfolio returns (standarize protfolio prices %)
        logs = (np.log(cl) - np.log(op)) * 100
        rets = (cl / op - 1) * 100
        vols = (hi / lo - 1) * 100
        higs = (hi / cl - 1) * 100
        lows = (cl / lo - 1) * 100

        return logs, rets, vols, higs, lows

    # get data by year
    def reduceDimension( self, logs, rets, vols, higs, lows ):

        ln = len(self.ccys)

        for ccy in self.ccys:

            logs_base = logs[logs.filter(regex=ccy + "_").columns].sum(axis=1)
            logs_term = (
                logs[logs.filter(regex="_" + ccy).columns]
                .apply(lambda x: -x)
                .sum(axis=1)
            )
            logs[ccy] = (logs_base + logs_term) / ln

            rets_base = rets[rets.filter(regex=ccy + "_").columns].sum(axis=1)
            rets_term = (
                rets[rets.filter(regex="_" + ccy).columns]
                .apply(lambda x: -x)
                .sum(axis=1)
            )
            rets[ccy] = (rets_base + rets_term) / ln

            vols[ccy] = vols[vols.filter(regex=ccy).columns].sum(axis=1) / ln
            higs[ccy] = higs[higs.filter(regex=ccy).columns].sum(axis=1) / ln
            lows[ccy] = lows[lows.filter(regex=ccy).columns].sum(axis=1) / ln

        logs = logs[self.ccys]
        rets = rets[self.ccys]
        vols = vols[self.ccys]
        higs = higs[self.ccys]
        lows = lows[self.ccys]

        idxs = pd.DataFrame(index=rets.index, columns=self.ccys)
        # create synthetic standarize idxs prices
        last_dt = 0
        for ccy in self.ccys:
            for i, dtime in enumerate(rets.index):
                if i == 0:
                    idxs[ccy][dtime] = 100
                    last_dt = dtime
                else:
                    idxs[ccy][dtime] = idxs[ccy][last_dt] * (
                        1 + rets[ccy][dtime] / 100
                    )
                    last_dt = dtime

        return logs, rets, vols, higs, lows, idxs

    # get currencies [str]
    def getCcys( self ):
        ccys = []
        for sym in self.symbols:
            ccy = sym.split("_")
            if ccy[0] not in ccys:
                ccys.append(ccy[0])
            if ccy[1] not in ccys:
                ccys.append(ccy[1])
        ccys.sort()
        return ccys

    ### INDICATORS ###

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
        Exponential movint average
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
            ema_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
            ema_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
            rs = ema_up / ema_down
            rsi = 100 - (100/(1 + rs))
            data[f'{currency}_rsi{periods}'] = rsi

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



if __name__ == "__main__":

    data = DataSet()

    op, hi, lo, cl, vo = data.getCandles()
    logs, rets, vols, higs, lows = data.normalizeData(op, hi, lo, cl, vo)
    logs_, rets_, vols_, higs_, lows_, idxs_ = data.reduceDimension(logs, rets, vols, higs, lows)



