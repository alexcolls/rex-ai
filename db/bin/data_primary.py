
# author: Quantium Rock
# license: MIT

import os
from pathlib import Path
from datetime import datetime
import pandas as pd

from apis.oanda_api import OandaApi
from config import SYMBOLS
from config import TIMEFRAME


### PrimaryData -> download candles from oanda_api to ../data/primary/<year>/:
# opens.csv, highs.csv, lows.csv, closes.csv, volumes.csv

class PrimaryData:

    ## class constructor
    def __init__ ( self, symbols=SYMBOLS, timeframe=TIMEFRAME, start_year=2005 ):
        
        # quotes granularity default=5_second_bars
        self.symbols = symbols
        self.timeframe = timeframe
        self.start_year = start_year
        self.primary_path = '../data/primary/'
        self.db_path = self.primary_path
        

    ## update data of primary db missing years & weeks
    def updateDB ( self ):

        start_time = datetime.utcnow()
        # if missing years download year
        if ( self.missing_years ):
            for year in self.missing_years:
                print(year)
                self.getData( year=year )
        
        print('\nDB updated!')
        final_time = datetime.utcnow()
        duration = final_time - start_time
        print('\nIt took', round(duration.total_seconds()/60), 'minutes to update the data.')

        return True
  

    ## check missing years in db/data/
    def checkDB ( self ):

        # check missing years since <start_year>
        missing_years = []
        #current_year = datetime.utcnow().year
        years = [ yr for yr in range(self.start_year, datetime.utcnow().year+1) ]

        # iterate each folder of db/data/primary/<year>/<week>/
        years_db = os.listdir(self.db_path)
        for year in years:
            if not str(year) in years_db:
                # append missing year
                print('Missing year:', year)
                missing_years.append(year)                    

        # if no asks_bids weeks missing
        if not missing_years:
            print('\nDB is fully updated since', self.start_year, '\n')

        self.missing_years = missing_years

        return True


    ##_ PrimaryData.getYear(2022)
        """ 
            1. download candles (mid)
            2. for each symbol in the portfolio, by default SYMBOLS
            3. from the first monday to the last friday[-2] of the year
            4. group all symbols data by weeks
            5. store each week locally into ../data/primary/ by year
         """
    def getData ( self, year=2022 ):

        oanda_api = OandaApi()

        start_date = str(year) + '-01-01'

        # init daily dataframes indices for asks & bids
        op = pd.DataFrame(index=[start_date])
        hi = pd.DataFrame(index=[start_date])
        lo = pd.DataFrame(index=[start_date])
        cl = pd.DataFrame(index=[start_date])
        vo = pd.DataFrame(index=[start_date])

        # iterate each instrument to get a full week data each
        for symbol in self.symbols:

            print(symbol)
 
            # initialize symbol data struct
            data = { 'dtime': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': [] }

            iterate = True
            # iterate until the end of year
            while iterate:

                # request 5000 bars from oanda rest-api
                req = oanda_api.getCandles( symbol, self.timeframe, start_date, include_frist=True )

                # iterate each candle
                for x in req:
                    # if current candle time changed year
                    if pd.to_datetime(x['time']).year > year:
                        iterate = False
                        break # close year

                    # append data
                    data['dtime'].append( x['time'] )
                    data['open'].append( float(x['mid']['o']) )
                    data['high'].append( float(x['mid']['h']) )
                    data['low'].append( float(x['mid']['l']) )
                    data['close'].append( float(x['mid']['c']) )
                    data['volume'].append( int(x['volume']) )
                
                if len(data['dtime']) > 0:
                # only for current year: check if there is no more history
                    if start_date == data['dtime'][-1]:
                        iterate = False
                        del req
                        break
                    # otherwise update start_date with last loop request
                    start_date = data['dtime'][-1]

            # ^ finished symbol year

            # transform data to prices dataframe
            dfs = []
            for i in list(data.keys())[1:]:
                df = pd.DataFrame(data[i], index=data['dtime'], columns=[symbol])
                #df.index = pd.to_datetime(df.index)
                dfs.append( df )

            op = pd.merge(op, dfs[0], how='outer', left_index=True, right_index=True)
            hi = pd.merge(hi, dfs[1], how='outer', left_index=True, right_index=True)
            lo = pd.merge(lo, dfs[2], how='outer', left_index=True, right_index=True)
            cl = pd.merge(cl, dfs[3], how='outer', left_index=True, right_index=True)
            vo = pd.merge(vo, dfs[4], how='outer', left_index=True, right_index=True)

            # realese memory
            del data, dfs

        # ^ finished all symbols

        # fill nans with forward-fill (last non-nan price)
        op.fillna(method='ffill', inplace=True)
        hi.fillna(method='ffill', inplace=True)
        lo.fillna(method='ffill', inplace=True)
        cl.fillna(method='ffill', inplace=True)

        # fill nans with backward-fill (for the first seconds of the week)
        op.fillna(method='bfill', inplace=True)
        hi.fillna(method='bfill', inplace=True)
        lo.fillna(method='bfill', inplace=True)
        cl.fillna(method='bfill', inplace=True)

        # fill volume nans with 0
        vo.fillna(0, inplace=True)

        # create path ../data/primary/<year>/
        out_path = self.primary_path + str(year) + '/'
        Path(out_path).mkdir(parents=True, exist_ok=True)

        # save daily csv into year week folder
        op.to_csv(out_path + 'opens.csv', index=True)
        hi.to_csv(out_path + 'highs.csv', index=True)
        lo.to_csv(out_path + 'lows.csv', index=True)
        cl.to_csv(out_path + 'closes.csv', index=True)
        vo.to_csv(out_path + 'volumes.csv', index=True)

        print('\n...saving successfully opens.csv, highs.csv, lows.csv, closes.csv, volumes.csv in', out_path, '\n')

        # realese memory
        del op, hi, lo, cl, vo

        return True

    ##__primarData.getYear()



    
    






 