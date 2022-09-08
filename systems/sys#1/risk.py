
import numpy as np
import pandas as pd
from data import DataSet
from config import RISK, BALANCE, LEVERAGE


class RiskManagement:


    def __init__( self, risk=RISK, balance=BALANCE, leverage=LEVERAGE ):

        self.risk = risk
        self.balance = balance
        self.leverage = leverage
        self.data = DataSet()
      # o, h, l, c  
        op, hi, lo, cl, _ = self.getCandles()
        logs, rets, vols, higs, lows = self.data.normalizeData(op, hi, lo, cl)
        self.logs_, _, _, _, _, self.idxs_ = self.data.reduceDimension(logs, rets, vols, higs, lows)
        self.predictions = self.data.makePredictions()


    def mean_volatility_prediction(self, dev=2, rate=120):

        data = pd.DataFrame([])
        for ccy in self.logs.columns:
            data[ccy] = np.abs(self.logs[ccy]).mean() + self.logs[ccy].std()*dev
        data.index = self.logs.index
        data2 = data.iloc[-1:].copy()

        return data, data2


    def correlation_pairs(self, pred_vols):

        data = pd.DataFrame(index=pred_vols.index)
        for pair in pred_vols.columns:
            for second_pair in pred_vols.columns:
                data[f'{pair}_{second_pair}_corr'] = pred_vols[pair].corr(pred_vols[second_pair])

        data.index = pred_vols.index
        data = data.iloc[-1:].T
        data.columns = ["correlation"]
        data = data[round(data["correlation"],6) != 1.000000]
        data = data.T

        return data


    def expected_volatility(self, pred_vol, correlations):

        vols = pred_vol.copy()
        pred_vol = abs(pred_vol)


        for sym in pred_vol.columns:

            real_vol = 0
            for sym2 in pred_vol.columns:

                if sym == sym2:
                    real_vol += pred_vol[sym]

                else:
                    real_vol += pred_vol[sym2] * correlations.filter(regex=sym+'_'+sym2).values[0][0]

            vols[sym] = abs(round(real_vol.mean(),6))

        total_vol = vols.mean(axis=1)[0]


        return total_vol, vols


    def weighted_volatility(self, volatility):
        weight = pd.DataFrame([])
        volatility = volatility.copy()
        volatility_T = volatility.T
        weight = volatility_T/ volatility_T.mean()

        return  weight


    def exchange_rates(self, df, acc_ccy='USD'):

        rate = df.iloc[-1:]

        data = pd.DataFrame(index=rate.index)

        for ccy in rate.columns:
            if ccy.split("_",2)[0] == acc_ccy:
                data[(ccy.split("_")[1])] = rate[ccy]
            elif ccy.split("_",2)[1] == acc_ccy:
                data[(ccy.split("_")[0])] = 1/rate[ccy]

        return data


    def trade_signals(self, pred_vols, logs, classification, exchange_rate ):

        cor = self.correlation_pairs(logs)

        t,v = self.expected_volatility(pred_vols, cor)
        n = int((abs(classification.T)).sum())
        risked = self.risk*self.balance*self.leverage/(t*n)

        trade_orders = []
        trade_dataframe = {"datetime":[],"currency":[], "side":[], "size":[]}
        for ccy in classification.columns:
            d = {}
            if int(classification[ccy]) != 0:
                d["currency"] = ccy
                trade_dataframe["currency"].append(ccy)
                signal = "sell" if int(classification[ccy]) == -1 else "buy"
                trade_dataframe["side"].append(signal)

                if ccy.split("_")[0] == "USD":
                    d["size"] = round(risked*int(classification[ccy]))
                    trade_dataframe["size"].append(round(risked))
                else:
                    d["size"] = round(risked*float(exchange_rate[ccy.split("_")[0]])*int(classification[ccy]))
                    trade_dataframe["size"].append(round(risked*float(exchange_rate[ccy.split("_")[0]])))
                d["datetime"] = v.index.to_list()[0].strftime("%Y-%m-%dT%H:%M:%S.Z")
                trade_dataframe["datetime"].append(v.index.to_list()[0])

                trade_orders.append(d)

        trade_dataframe = pd.DataFrame.from_dict(trade_dataframe)
        trade_dataframe.set_index("datetime",inplace=True)
        trade_dataframe.index = pd.to_datetime(trade_dataframe.index)

        return trade_orders, trade_dataframe


if __name__ == "__main__":

    rm = RiskManagement()

    data.makePredictions()