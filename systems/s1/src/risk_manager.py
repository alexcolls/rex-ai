
# author: Marti Llanes & Quantium Rock
# license: MIT

import json
import numpy as np
import pandas as pd
from predictions import Predictions
from account import Account

with open('config.json') as json_file:
    config = json.load(json_file)

RISK = config['RISK']
BALANCE = config['BALANCE']
LEVERAGE = config['LEVERAGE']


class RiskManager( Account, Predictions ):

    def __init__( self, risk=RISK, balance=BALANCE, leverage=LEVERAGE ):

        super().__init__()
        self.risk = risk
        self.balance = balance
        self.leverage = leverage
        self.volatilities, _ = self.mean_volatility_prediction()
        self.correlations = self.correlation_pairs(self.logs)
        self.variance, self.exp_volatilies = self.expected_volatility(self.volatilities, self.correlations)
        self.wei_volatilities = self.weighted_volatility(self.exp_volatilies)
        self.fx_rates = self.exchange_rates(self.fx_rates)


    def mean_volatility_prediction(self, n_devs=2 ):

        data = pd.DataFrame([])
        for ccy in self.logs.columns:
            data[ccy] = np.abs(self.logs[ccy]).mean() + self.logs[ccy].std()*n_devs
        data.index = self.logs.index
        data2 = data.iloc[-1:].copy()

        return data, data2


    def correlation_pairs(self, logs):

        data = pd.DataFrame(index=logs.index)
        for pair in logs.columns:
            for second_pair in logs.columns:
                data[f'{pair}_{second_pair}_corr'] = logs[pair].corr(logs[second_pair])

        data.index = logs.index
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
        weight = volatility_T/volatility_T.mean()

        return  weight


    

    
    def makePositions( self ):
        
        for x in self.predictions:
            pass
            

    def exchange_rates(self, df, units):

        rate = df.iloc[-1:]

        data = pd.DataFrame(index=rate.index)

        acc_ccy = self.account['ccy']

        for ccy in self.symbols:
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

    rm = RiskManager()

