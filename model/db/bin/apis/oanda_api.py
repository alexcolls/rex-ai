
# OANDA BROKER REST API

# author: Quantium Rock
# date: August 2022
# license: MIT

# dependencies
import requests
import json


# API CLIENT

class OandaApi:

    # url constants
    ENVIRONMENTS = {
        # demo account
        "no_trading": {
            "stream": 'https://stream-fxpractice.oanda.com',
            "api": 'https://api-fxpractice.oanda.com'
        },
        # real account
        "live": {
            "stream": 'https://stream-fxtrade.oanda.com',
            "api": 'https://api-fxtrade.oanda.com'
        }
    }

    def __init__  ( self, PRIVATE_KEY=False, LIVE_TRADING=False ):

        # upload Onada authenthification ./keys/oanda.json
        with open("db/bin/apis/keys/oanda.json") as x:
            self.__auth__ = json.load(x)

        self.TOKEN = self.__auth__['PUBLIC_TOKEN']
        if PRIVATE_KEY:
            self.TOKEN = self.__auth__['PRIVATE_TOKEN']
        
        # set trading enviroment
        self.enviroment = self.ENVIRONMENTS['no_trading']['api']
        if LIVE_TRADING:
            self.enviroment = self.ENVIRONMENTS['live']['api']

        # set request session  and add authentification metadata
        self.client = requests.Session()
        self.client.headers['Authorization'] = 'Bearer '+ self.TOKEN


    #__ OandaApi.getRawCandles('EUR_USD', 'H1', '2022-01-01')
    
    def getRawCandles ( self, symbol, timeframe, start_date, count=5000, include_frist=False, api_version='v3' ):

        try:
            req = self.client.get(f'{self.enviroment}/{api_version}/instruments/{symbol}/candles?count={count}&price=BA&granularity={timeframe}&from={start_date}&includeFirst={include_frist}')

            return json.loads(req.content.decode('utf-8'))['candles']
        except:
            return 0


    # ORDER EXECUTOR

        # TODO


# END