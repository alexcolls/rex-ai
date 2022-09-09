# author: Marti Llanes & Quantium Rock
# license: MIT

import json
import numpy as np
from src.predictions import Predictions
from src.account import Account

with open('config.json') as json_file:
    config = json.load(json_file)

BALANCE = config['BALANCE']
RISK = config['RISK_ALLOWANCE']
LEVERAGE = config['MAX_LEVERAGE']


class RiskManager( Account, Predictions ):

    def __init__( self, risk=RISK, balance=BALANCE, leverage=LEVERAGE ):

        Account.__init__(self)
        Predictions.__init__(self)

        self.risk = risk
        self.balance = balance
        self.leverage = leverage
        self.market_volatility, self.sym_vols = self.marketVolatility(self.logs)
        self.correlations = self.logs.corr()
        self.variance, self.exp_volatilies = self.expectedVolatility()
        self.wei_vols = self.weightedVolatilities()
        self.new_orders = self.makeOrders()


    def marketVolatility( self, df, devs=2 ):

        sym_vols = {}
        for sym in df.columns:
            sym_vols[sym] = round(np.abs(df[sym]).mean() + df[sym].std() * devs, 4)

        tot_vol = round(np.abs(df[sym]).mean() + df[sym].std() * devs, 4)

        return tot_vol, sym_vols


    def expectedVolatility( self ):

        exp_vols = {}
        for sym1 in self.sym_vols.keys():
            cum_vol = 0
            for sym2 in self.sym_vols.keys():
                if sym1 == sym2:
                    cum_vol += self.sym_vols[sym1]
                else:
                    try:
                        cum_vol += self.sym_vols[sym2] * self.correlations[sym1][sym2]
                    except:
                        cum_vol += self.sym_vols[sym2] * self.correlations[sym2][sym1]
            
            exp_vols[sym1] = abs(round(cum_vol.mean(),6))

        vols = [ x for x in exp_vols.values() ]
        variance = round(np.mean(vols, axis=0) + np.std(vols, axis=0), 4)

        return variance, exp_vols


    def weightedVolatilities( self ):

        wei_vols = {}
        for sym in self.sym_vols.keys():
            wei_vols[sym] = round(self.exp_volatilies[sym]/self.variance, 2)

        return  wei_vols


    def makeOrders( self ):

        self.risk_ratio = round(self.risk / (self.variance/100), 2)
        self.margin = self.account_state['NAV'] * self.risk_ratio * self.leverage

        new_orders = {}
        for sym, p in self.predictions.items():
            if p == 0:
                new_orders[sym] = 0
            else:
                units = int(self.margin * p * self.wei_vols[sym])
                if sym[:3] != self.account_state['ccy']:
                    try:
                        units = int(units * self.fx_rates[sym[:3]+'_'+self.account_state['ccy']])
                    except:
                        units = int(units / self.fx_rates[self.account_state['ccy']+'_'+sym[:3]])
                
                new_orders[sym] = units

        return new_orders
            

    def stopOut( self, minutes=15 ):

        return 0


if __name__ == "__main__":

    rm = RiskManager()

    rm.makePositions()


# end