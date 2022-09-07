# OANDA BROKER REST API

# author: Quantium Rock
# date: August 2022
# license: MIT

# dependencies
import requests
import json
import os


# API CLIENT


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

        # upload Onada authenthification ./keys/oanda.json
        APIKEY_PATH = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "oanda_key.json")
        )
        with open(APIKEY_PATH) as x:
            self.__auth__ = json.load(x)

        self.TOKEN = self.__auth__["PUBLIC_TOKEN"]
        if PRIVATE_KEY:
            self.TOKEN = self.__auth__["PRIVATE_TOKEN"]

        # set trading enviroment
        self.enviroment = self.ENVIRONMENTS["no_trading"]["api"]
        if LIVE_TRADING:
            self.enviroment = self.ENVIRONMENTS["live"]["api"]

        # set request session  and add authentification metadata
        self.client = requests.Session()
        self.client.headers["Authorization"] = "Bearer " + self.TOKEN
        
        self.api_version = 'v3'

        self.accounts = self.getAccounts()

    
    # return a list of all the api accounts ids
    def getAccounts( self ):

        req = 0
        try:
            req = self.client.get(
                f"{self.enviroment}/{self.api_version}/accounts"
            )
            req = json.loads(req.content.decode("utf-8"))['accounts']
        except:
            print("OANDA API ERROR", Exception)
            return

        accs = []
        for x in req:
            accs.append(x['id'])
        
        return accs

     
    # return the account state (NAV, PnL, margin, accCurrency, etc)
    def getSummary( self, account_id ):
        
        if account_id is None:
            account_id = self.accounts[0]

        req = 0
        try:
            req = self.client.get(
                f"{self.enviroment}/{self.api_version}/accounts/{account_id}/summary"
            )
            return json.loads(req.content.decode("utf-8"))['accounts']
        except:
            print("OANDA API ERROR", Exception)

    
    # return a json with all tradeable instruments for a given accoount
    def getInstruments( self, account_id ):
        
        if account_id is None:
            account_id = self.accounts[0]

        req = 0
        try:
            req = self.client.get(
                f"{self.enviroment}/{self.api_version}/accounts/{account_id}/instruments"
            )
            return json.loads(req.content.decode("utf-8"))['instruments']
        except:
            print("OANDA API ERROR", Exception)

    
    # return a list of the current trading positions for a given account
    def getOpenPositions( self, account_id ):
        
        if account_id is None:
            account_id = self.accounts[0]

        req = 0
        try:
            req = self.client.get(
                f"{self.enviroment}/{self.api_version}/accounts/{account_id}/openPositions"
            )
            return json.loads(req.content.decode("utf-8"))['positions']
        except:
            print("OANDA API ERROR", Exception)


    
    # return a list with the PnLs of each instrument for a given account
    def getAllPositions( self, account_id ):
        
        if account_id is None:
            account_id = self.accounts[0]

        req = 0
        try:
            req = self.client.get(
                f"{self.enviroment}/{self.api_version}/accounts/{account_id}/positions"
            )
            return json.loads(req.content.decode("utf-8"))['positions']
        except:
            print("OANDA API ERROR", Exception)


    # return a list of the current open trades for a given account
    def getOpenTrades( self, account_id ):
        
        if account_id is None:
            account_id = self.accounts[0]

        req = 0
        try:
            req = self.client.get(
                f"{self.enviroment}/{self.api_version}/accounts/{account_id}/openTrades"
            )
            return json.loads(req.content.decode("utf-8"))['trades']
        except:
            print("OANDA API ERROR", Exception)


    # return a list of all historical trades of the account
    def getAllTrades( self, account_id ):
        
        if account_id is None:
            account_id = self.accounts[0]

        req = 0
        try:
            req = self.client.get(
                f"{self.enviroment}/{self.api_version}/accounts/{account_id}/openTrades"
            )
            return json.loads(req.content.decode("utf-8"))['trades']
        except:
            print("OANDA API ERROR", Exception)


    # return json with history candles between 2 dates (max periods=5000)
    def getCandles( self, symbol, timeframe, start_date, count=5000, include_frist=False, mids=True ):

        prices = "M" if mids else "BA"  # Mids or BidAsks

        try:
            req = self.client.get(
                f"{self.enviroment}/{self.api_version}/instruments/{symbol}/candles?count={count}&price={prices}&granularity={timeframe}&from={start_date}&includeFirst={include_frist}"
            )

            return json.loads(req.content.decode("utf-8"))["candles"]
        except:
            print("OANDA API ERROR", Exception)
    

    # ORDER EXECUTOR

    def postOrder ( self ):
        return


# END
