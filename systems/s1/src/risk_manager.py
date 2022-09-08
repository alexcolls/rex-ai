# author: Marti Llanes & Quantium Rock
# license: MIT

import json
import numpy as np
import pandas as pd
from predictions import Predictions
from account import Account

with open('config.json') as json_file:
    config = json.load(json_file)

BALANCE = config['BALANCE']
RISK = config['RISK_ALLOWANCE']
LEVERAGE = config['MAX_LEVERAGE']


class RiskManager( Account, Predictions ):

    def __init__( self, risk=RISK, balance=BALANCE, leverage=LEVERAGE ):

        super(Account, self).__init__()
        super(Predictions, self).__init__()

        self.risk = risk
        self.balance = balance
        self.leverage = leverage
        self.volatilities, _ = self.mean_volatility_prediction(self.logs)
        self.correlations = self.correlation_pairs(self.logs)
        self.variance, self.exp_volatilies = self.expected_volatility(self.volatilities, self.correlations)
        self.wei_volatilities = self.weighted_volatility(self.exp_volatilies)


    def mean_volatility_prediction(self, logs, n_devs=2 ):

        data = pd.DataFrame([])
        for ccy in logs.columns:
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
        
        for sym, x in self.predictions.iteritems():
            print(sym, x)
            


if __name__ == "__main__":

    rm = RiskManager()

    rm.makePositions()

