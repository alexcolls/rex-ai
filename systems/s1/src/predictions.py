# author: Quantium Rock & Marti Llanes
# license: MIT

import os
import json
import pickle
import numpy as np
import pandas as pd
from keras.models import load_model
from libs.oanda_api import OandaApi
from libs.indicators import Indicators

with open('config.json') as json_file:
    config = json.load(json_file)

SYMBOLS = config['SYMBOLS'] 
TIMEFRAME = config['TIMEFRAME']
LOOKBACK = config['LOOKBACK']


class Predictions:

    def __init__(self, symbols=SYMBOLS, timeframe=TIMEFRAME, lookback=LOOKBACK):

        # quotes granularity
        self.symbols = symbols
        self.timeframe = timeframe
        self.lookback = lookback
        self.ccys = self.getCcys()
        self.db_path = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/series/")
        )

    # get currencies [str]
    def getCcys(self):
        ccys = []
        for sym in self.symbols:
            ccy = sym.split("_")
            if ccy[0] not in ccys:
                ccys.append(ccy[0])
            if ccy[1] not in ccys:
                ccys.append(ccy[1])
        ccys.sort()
        return ccys

    # get data by year
    def getCandles(self):

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
    def normalizeData(self, op, hi, lo, cl):

        # create portfolio returns (standarize protfolio prices %)
        logs = (np.log(cl) - np.log(op)) * 100
        rets = (cl / op - 1) * 100
        vols = (hi / lo - 1) * 100
        higs = (hi / cl - 1) * 100
        lows = (cl / lo - 1) * 100

        return logs, rets, vols, higs, lows

    # get data by year
    def reduceDimension(self, logs, rets, vols, higs, lows):

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
        # create synthetic standarize idxs prices (last_price+last_price*current_return)
        last_dt = 0
        for ccy in self.ccys:
            for i, dtime in enumerate(rets.index):
                if i == 0:
                    idxs[ccy][dtime] = 100
                    last_dt = dtime
                else:
                    idxs[ccy][dtime] = idxs[ccy][last_dt] * (1 + rets[ccy][dtime] / 100)
                    last_dt = dtime

        return logs, rets, vols, higs, lows, idxs


    # make indicators
    def makeIndicators( self, df ):

        indics = Indicators()
        time = indics.time_standard(df)
        lp = indics.lowpass_filter(df)
        lp_m = indics.lowpass_momentum(df)
        hp = indics.highpass_filter(df)
        hp_m = indics.highpass_momentum(df)
        r8 = indics.rsi(df, window=8)
        r24 = indics.rsi(df, window=24)
        r120 = indics.rsi(df, window=120)
        e8 = indics.ema(df, window=8)
        e24 = indics.ema(df, window=24)
        e120 = indics.ema(df, window=120)
        ed8 = indics.ema_diff(df, window=8)
        ed24 = indics.ema_diff(df, window=24)
        ed120 = indics.ema_diff(df, window=120)
        sr = indics.sharpe_ratio(df, window=24)
        bb = indics.bollinger_bands(df, window=24, n_devs=[1])

        data = lp.join(hp)
        data = data.join(lp_m)
        data = data.join(hp_m)
        data = data.join(r8)
        data = data.join(r24)
        data = data.join(r120)
        data = data.join(e8)
        data = data.join(e24)
        data = data.join(e120)
        data = data.join(sr)
        data = data.join(bb)
        data = data.join(ed8)
        data = data.join(ed24)
        data = data.join(ed120)
        data = df.join(data)
        data = data.join(time, how="outer")

        return data

    
    def randomPredictions( self ):

        predictions = {}
        for sym in self.symbols:
            predictions[sym] = np.random.randint(-1, 2, 1)[0]

        return predictions


    def makePredictions( self ):

        op, hi, lo, cl, vo = self.getCandles()

        print(cl)

        logs, rets, vols, higs, lows = self.normalizeData(op, hi, lo, cl)

        print(logs)

        logs_, rets_, vols_, higs_, lows_, idxs_ = self.reduceDimension(logs, rets, vols, higs, lows)

        print(idxs_)

        indicators = self.makeIndicators(df=idxs_)

        predictions = {}
        for sym in self.symbols:
            
            X = indicators.filter(regex=f"{sym[:3]}|{sym[4:]}|sin|cos").copy()

            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            X.fillna(method='bfill', inplace=True)
            X.fillna(method='ffill', inplace=True)

            with open(f'model/s4_{sym}.pkl' , 'rb') as pickle_file:
                scaler = pickle.load(pickle_file)
            X = scaler.transform(X)

            print(X)

            # make sequences and output tensors
            def makeSequences( X, lookback=LOOKBACK ):

                X_tensor = []
                for i in range(lookback, X.shape[0]):
                    try:
                        X_tensor.append(X.iloc[ i-lookback : i ])
                    except:
                        break

                return np.array(X_tensor)

            X_pred = makeSequences( X )

            print(X_pred.shape)

            X_pred = np.expand_dims(X_pred, axis=1)

            print(X_pred.shape)

            model = load_model(f"model/m4_{sym}.h5")

            predictions[sym] = model.predict(X_pred)

        print(predictions)

        return pd.DataFrame.from_dict(predictions)



if __name__ == "__main__":

    data = Predictions().randomPredictions()

    print(data)


# end