
### MODEL MAIN CONFIGURATION

# if you modify this file make sure to delete your database and download/update data again

# a) select trading universe:

SYMBOLS = [ 'AUD_CAD', 'AUD_CHF', 'AUD_JPY', 'AUD_NZD', 'AUD_USD', 
            'CAD_CHF', 'CAD_JPY', 'CHF_JPY', 'EUR_AUD', 'EUR_CAD', 
            'EUR_CHF', 'EUR_GBP', 'EUR_JPY', 'EUR_NZD', 'EUR_USD', 
            'GBP_AUD', 'GBP_CAD', 'GBP_CHF', 'GBP_JPY', 'GBP_NZD', 
            'GBP_USD', 'NZD_CAD', 'NZD_CHF', 'NZD_JPY', 'NZD_USD', 
            'USD_CAD', 'USD_CHF', 'USD_JPY'
          ]


# b) model granularity (S5, M1, M15, H1, H8, D)

TIMEFRAME = 'M1'