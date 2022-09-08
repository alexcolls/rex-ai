# OANDA BROKER REST API

# author: Quantium Rock
# date: August 2022
# license: MIT

import requests
import json
import os

# api client
class OandaApi:

    # url constants
    ENVIRONMENTS = {
        # demo account
        "no_trading": {
            "stream": "https://stream-fxpractice.oanda.com",
            "api": "https://api-fxpractice.oanda.com",
        },
        # real account
        "live": {
            "stream": "https://stream-fxtrade.oanda.com",
            "api": "https://api-fxtrade.oanda.com",
        },
    }

    def __init__( self, PRIVATE_KEY=False, LIVE_TRADING=False ):

        # upload Onada authenthification secret key
        APIKEY_PATH = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../config.json")
        )
        with open(APIKEY_PATH) as config:
            self.__auth__ = json.load(config)

        self.TOKEN = self.__auth__["OANDA_KEY"]

        # set trading enviroment
        self.enviroment = self.ENVIRONMENTS["no_trading"]["api"]
        if LIVE_TRADING:
            self.enviroment = self.ENVIRONMENTS["live"]["api"]

        # set request session  and add authentification metadata
        self.client = requests.Session()
        self.client.headers["Authorization"] = "Bearer " + self.TOKEN
        self.client.headers['Content-Type'] = 'application/json'      
        self.api_version = 'v3'
        self.accounts = self.getAccounts()


    ### GET methods ###

    # return a list of all the api accounts ids
    def getAccounts( self ):

        req = 0
        try:
            req = self.client.get(
                f"{self.enviroment}/{self.api_version}/accounts"
            )
            req = json.loads(req.content.decode("utf-8"))['accounts']
        except Exception as e:
            print(e)

        accs = []
        for x in req:
            accs.append(x['id'])
        
        return accs

     
    # return the account state (NAV, PnL, margin, accCurrency, etc)
    def getSummary( self, account_id=None ):
        
        if account_id is None:
            account_id = self.accounts[0]

        try:
            req = self.client.get(
                f"{self.enviroment}/{self.api_version}/accounts/{account_id}/summary"
            )
            return json.loads(req.content.decode("utf-8"))['account']
        except Exception as e:
            print(e)

    
    # return a json with all tradeable instruments for a given accoount
    def getInstruments( self, account_id ):
        
        if account_id is None:
            account_id = self.accounts[0]

        try:
            req = self.client.get(
                f"{self.enviroment}/{self.api_version}/accounts/{account_id}/instruments"
            )
            return json.loads(req.content.decode("utf-8"))['instruments']
        except Exception as e:
            print(e)

    
    # return a list of the current trading positions for a given account
    def getOpenPositions( self, account_id ):
        
        if account_id is None:
            account_id = self.accounts[0]

        try:
            req = self.client.get(
                f"{self.enviroment}/{self.api_version}/accounts/{account_id}/openPositions"
            )
            return json.loads(req.content.decode("utf-8"))['positions']
        except Exception as e:
            print(e)

    
    # return a list with the PnLs of each instrument for a given account
    def getAllPositions( self, account_id ):
        
        if account_id is None:
            account_id = self.accounts[0]

        try:
            req = self.client.get(
                f"{self.enviroment}/{self.api_version}/accounts/{account_id}/positions"
            )
            return json.loads(req.content.decode("utf-8"))['positions']
        except Exception as e:
            print(e)


    # return a list of the current open trades for a given account
    def getOpenTrades( self, account_id ):
        
        if account_id is None:
            account_id = self.accounts[0]

        try:
            req = self.client.get(
                f"{self.enviroment}/{self.api_version}/accounts/{account_id}/openTrades"
            )
            return json.loads(req.content.decode("utf-8"))['trades']
        except Exception as e:
            print(e)


    # return a list of all historical trades of the account
    def getAllTrades( self, account_id ):
        
        if account_id is None:
            account_id = self.accounts[0]

        try:
            req = self.client.get(
                f"{self.enviroment}/{self.api_version}/accounts/{account_id}/openTrades"
            )
            return json.loads(req.content.decode("utf-8"))['trades']
        except Exception as e:
            print(e)

    
    # return a list of the current pending orders for a given account
    def getPendingOrders( self, account_id ):
        
        if account_id is None:
            account_id = self.accounts[0]

        try:
            req = self.client.get(
                f"{self.enviroment}/{self.api_version}/accounts/{account_id}/pendingOrders"
            )
            return json.loads(req.content.decode("utf-8"))['orders']
        except Exception as e:
            print(e)
            

    # return a list of all historical orders of the account
    def getAllOrders( self, account_id ):
        
        if account_id is None:
            account_id = self.accounts[0]

        try:
            req = self.client.get(
                f"{self.enviroment}/{self.api_version}/accounts/{account_id}/openTrades"
            )
            return json.loads(req.content.decode("utf-8"))['trades']
        except Exception as e:
            print(e)


    # return json with history candles between 2 dates (max periods=5000)
    # for a given instruments
    def getCandles( self, symbol, timeframe, start_date, count=5000, include_frist=False, mids=True ):

        prices = "M" if mids else "BA"  # Mids or BidAsks

        try:
            req = self.client.get(
                f"{self.enviroment}/{self.api_version}/instruments/{symbol}/candles?count={count}&price={prices}&granularity={timeframe}&from={start_date}&includeFirst={include_frist}"
            )

            return json.loads(req.content.decode("utf-8"))["candles"]
        except Exception as e:
            print(e)
    

    # return last n candles by symbol & timeframe
    def getLastCandles( self, symbol, timeframe, count=5000, mids=True ):

        prices = "M" if mids else "BA"  # Mids or BidAsks

        try:
            req = self.client.get(
                f"{self.enviroment}/{self.api_version}/instruments/{symbol}/candles?count={count}&price={prices}&granularity={timeframe}"
            )

            return json.loads(req.content.decode("utf-8"))["candles"]
        except Exception as e:
            print(e)
    

    ### POST methods ###

    # open new order on a specific instrument & account
    def postOrder ( self, account_id, instrument, units, order_type='MARKET', time_in_force='FOK' ):

        order = { "order": {
                "type": order_type,
                "positionFill": "DEFAULT",
                "instrument": instrument,
                "timeInForce": time_in_force,
                "units": str(units)
                } }
                
        order = json.dumps(order, indent=4) 

        try:
            req = self.client.post( f"{self.enviroment}/{self.api_version}/accounts/{account_id}/orders", data=str(order) )
            return json.loads(req.content.decode("utf-8"))
        except Exception as e:
            print(e)


#__OandaApi()
