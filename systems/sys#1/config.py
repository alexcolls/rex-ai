
### TRADING SYSTEM MAIN CONFIGURATION ###

# select trading portfolio
SYMBOLS = [ 'AUD_CAD', 'AUD_CHF', 'AUD_JPY', 'AUD_NZD', 'AUD_USD', 
            'CAD_CHF', 'CAD_JPY', 'CHF_JPY', 'EUR_AUD', 'EUR_CAD', 
            'EUR_CHF', 'EUR_GBP', 'EUR_JPY', 'EUR_NZD', 'EUR_USD', 
            'GBP_AUD', 'GBP_CAD', 'GBP_CHF', 'GBP_JPY', 'GBP_NZD', 
            'GBP_USD', 'NZD_CAD', 'NZD_CHF', 'NZD_JPY', 'NZD_USD', 
            'USD_CAD', 'USD_CHF', 'USD_JPY'
          ]

# model granularity (S5, M1, M15, H1, H8, D)
TIMEFRAME = 'H1'

# time periods length
LOOKBACK = 24

# trading account number
ACCOUNT = '101-004-17169350-002'

# trading enviroment
ENVIROMENT = 'demo'

# risk management params

RISK = 0.01
BALANCE = 100000
LEVERAGE = 1
