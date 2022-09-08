# author: Quantium Rock
# license: MIT

import json
from datetime import datetime
from src.libs.oanda_api import OandaApi

with open('config.json') as json_file:
    config = json.load(json_file)

ACCOUNT = config['ACCOUNT']


class Account:

    def __init__( self, account_id=ACCOUNT ):
        
        self.oa = OandaApi()
        self.account_id = account_id
        self.account = self.getAccount()
        self.positions = self.getPositions()
        self.exposures = self.getExposures()
    

    def getAccount( self ):

        try:
            req = self.oa.getSummary(self.account_id)
        except:
            return 0

        account = {}
        account['dt'] = req['createdTime']
        account['id'] = req['id']
        account['ccy'] = req['currency']
        account['balance'] = round(float(req['balance']),2)
        account['close_pnl'] = round(float(req['resettablePL']),2)
        account['NAV'] = round(float(req['NAV']),2)
        account['open_pnl'] = round(float(req['unrealizedPL']),2)
        account['commissions'] = round(float(req['financing']),2)
        account['margin_used'] = round(float(req['marginUsed']),2)
        account['open_position'] = round(float(req['positionValue']),2)
        account['leverage'] = round(account['open_position'] / account['NAV'],2)

        return account


    def getPositions( self ):

        try:
            req = self.oa.getOpenPositions(self.account_id)
        except:
            return 0

        positions = []
        for x in req:
            units = 0
            price = 0
            if int(x['long']['units']) > 0:
                units = int(x['long']['units'])
                price = float(x['long']['averagePrice'])
            elif int(x['short']['units']) > 0:
                units = int(x['short']['units'])
                price = float(x['long']['averagePrice'])
            
            pos = { 'dt': datetime.strftime(datetime.utcnow(), '%Y-%m-%dT%H:%M:%S.%fZ'),
                    'symbol': x['instrument'],
                    'units': units,
                    'price': price,
                    'margin': round(float(x['marginUsed']),2),
                    'allocation': round(float(x['marginUsed'])/self.account['NAV'],2),
                    'pnl_close': round(float(x['pl']),2),
                    'pnl_open': round(float(x['unrealizedPL']),2),
                    'fees': round(float(x['financing'])+float(x['commission']),2)
                }

            positions.append(pos)
        
        return positions


    def getExposures( self ):

        if len(self.positions) == 0:
            return 0
        
        exposures = { 'dt': datetime.strftime(datetime.utcnow(), '%Y-%m-%dT%H:%M:%S.%fZ') }
        for pos in self.positions:

            base = pos['symbol'][:3]
            term = pos['symbol'][4:]

            side = 0
            if pos['units'] > 0:
                side = 1
            elif pos['units'] < 0:
                side = -1

            exposures[base] = round(pos['margin'] * 100 * side, 2)
            exposures[term] = round(pos['margin'] * 100 * -side, 2)
            
        return exposures


if __name__ == "__main__":

    acc = Account()

    print('\n\n> Account:', acc.account, '\n\n> Positions:', acc.positions, '\n\n> Exposures:', acc.exposures)

# end