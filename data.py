
# author: Quantium Rock
# license: MIT

import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from datetime import timedelta

# import Oanda api & instruments
from model.db.bin.apis.oanda_api import OandaApi
from model.config import SYMBOLS
from model.config import TIMEFRAME

from sys import argv

FIRST_YEAR = int(argv[1])

### PrimaryData -> ../data/primary/<year>/<week>/:
#   asks.csv, bids.csv, mids.csv, spreads.csv, volumes.csv
class PrimaryData:

    ## class constructor
    def __init__ ( self, start_year=FIRST_YEAR, symbols=SYMBOLS, timeframe=TIMEFRAME ):
        
        # quotes granularity default=5_second_bars
        self.symbols = symbols
        self.timeframe = timeframe
        self.start_year = int(start_year)
        self.primary_path = 'db/data/primary/'
        self.db_path = self.primary_path
        

    ## update data of primary db missing years & weeks
    def updateDB ( self ):

        start_time = datetime.utcnow()
        # if missing years download year
        if ( self.missing_years ):
            for year in self.missing_years:
                try:
                    print('\nDownloading data from', year, '...be patient, it can take a while...\n')
                    self.getData( year=year )
                except:
                    break
                                    
        # if missing weeks download weeks
        if ( self.missing_weeks ):
            for year, weeks in self.missing_weeks.items():
                for week in weeks:
                    try:
                        print('\nDownloading week', week, 'from', year, '\n')
                        self.getData( year=year, start_week=week, end_week=week )
                    except:
                        break
        
        print('\nDB updated!')

        final_time = datetime.utcnow()
        duration = final_time - start_time
        print('\nIt took', round(duration.total_seconds()/60/60, 2), 'hours to update the data.')

        return True
  

    ## check missing weeks & years in data/asks_bids since <year>
    def checkDB ( self ):

        # check missing years since <start_year>
        missing_years = []
        current_year = datetime.utcnow().year
        years = [ yr for yr in range(self.start_year, current_year+1) ]

        # init year missing weeks
        missing_weeks = {}

        # get current week of the year
        current_week = datetime.strptime(str(current_year)+'-01-01', '%Y-%m-%d')
        current_week = ( datetime.utcnow() - current_week ) / timedelta(weeks=1)
        current_week = int(current_week)

        # iterate each folder of db/data/primary/<year>/<week>/
        years_db = os.listdir(self.db_path)
        for year in years:
            if not str(year) in years_db:
                # append missing year
                print('Missing year:', year)
                missing_years.append(year)
            else: # if year exsits
                weeks_db = os.listdir(self.db_path + str(year))
                # if current year
                wks = current_week+2 if year == current_year else 52
                missing_weeks[year] = []
                for week in range(1, wks):
                    if not str(week) in weeks_db:
                        # append missing week
                        print('Missing week:', week, 'from', year)
                        missing_weeks[year].append(week)                              

        # delete empty keys
        missing_weeks = dict( [ (k,v) for k, v in missing_weeks.items() if len(v) > 0 ] )

        # if no asks_bids weeks missing
        if not missing_weeks and  not missing_years:
            print('\nDB is fully updated since', self.start_year, '\n')

        self.missing_years = missing_years
        self.missing_weeks = missing_weeks
        self.current_week = current_week


    ##_ PrimaryData.getData(2022)
        """ 
            1. download candles (bid/ask), min granularity = 5 seconds
            2. for each symbol in the portfolio, by default __universe__.SYMBOLS
            3. from the first monday to the last friday[-2] of the year
            4. group all symbols data by weeks
            5. store each week locally into ../data/primary/
         """
    def getData ( self, year=2022, start_week=1, end_week=51 ):

        oanda_api = OandaApi()

        # get first monday of the year
        first_monday = timedelta(days=( 7 - datetime.strptime(str(year), '%Y').weekday()))
        first_monday = datetime.strptime(str(year), '%Y') + first_monday

        # get all mondays of the year, except last one
        mondays = [ first_monday + timedelta(weeks=i) for i in range(start_week-1, end_week) ]

        # check mondays dates and add utc timezone
        for i, monday in enumerate(mondays):
            if monday.weekday() == 0:
                mondays[i] = datetime.strftime(monday, '%Y-%m-%d') + 'T00:00:00.000000000Z'
            else:
                raise Exception('ERROR: Its not a Monday!', i)

        wk = start_week-1 # init weeks counter
        
        # iterate on each monday of the year (weeks)
        for monday in mondays:

            wk += 1 # sum 1 week

            # add range frequency to each week
            freq = '1min'
            if self.timeframe.startswith('S') or self.timeframe.startswith('H'):
                freq = self.timeframe[::-1]

            # get each datetime of the week by timeframe ( default=5seconds )
            day_timestamps = pd.date_range(pd.to_datetime(monday), pd.to_datetime(monday) + timedelta(days=5), freq=freq)[:-1]

            # init daily dataframes indices for asks & bids
            asks = pd.DataFrame(index=day_timestamps)
            bids = pd.DataFrame(index=day_timestamps)
            vols = pd.DataFrame(index=day_timestamps)

            # iterate each __universe__.SYMBOL to get a full week data each
            for symbol in self.symbols:

                # print symbol and week of the year
                print(symbol)

                # initialize symbol variable
                data = { 'dt': [], 'ask': [], 'bid': [], 'volume': []}
                start_date = monday
                iterate = True

                # iterate until the end of week
                while iterate:

                    # request 5000 5-second-bars from oanda rest-api
                    req = oanda_api.getRawCandles( symbol, self.timeframe, start_date )

                    # print first date and last date of the request
                    try:
                        print(req[0]['time'][:19], req[-1]['time'][:19])
                    except: pass

                    # get the number of decimals of instrument's price
                    digits = 5
                    try:
                        digits = len(req[0]['ask']['c'].split('.')[1])
                    except: pass

                    # iterate each candle
                    for x in req:
                        # if current candle time changed week
                        if pd.to_datetime(x['time']) > asks.index[-1]:
                            iterate = False
                            break # close week

                        # append to data bid/ask quotes by datetime
                        data['dt'].append(x['time'])
                        # iterate bid/ask
                        for p in ['ask', 'bid']:
                            o = float(x[p]['o']) # open
                            h = float(x[p]['h']) # high
                            l = float(x[p]['l']) # low
                            c = float(x[p]['c']) # close
                            # round price = ( o + h + l + 2c ) * 0.2
                            #data[p].append( round(( o + h + l + 2*c ) / 5, digits) )
                            data[p].append(c)

                        # append period trading raw volume
                        data['volume'].append(int(x['volume']))
                    
                    if len(data['dt']) > 0:
                    # only for current year: check if there is no more history
                        if start_date == data['dt'][-1]:
                            iterate = False
                            del req
                            break
                        # otherwise update start_date with last loop request
                        start_date = data['dt'][-1]

                # ^ finished symbol week

                # transform data to asks dataframe
                _asks = pd.DataFrame(data['ask'], index=data['dt'], columns=[symbol])
                _asks.index = pd.to_datetime(_asks.index)
                
                # transform data to bids dataframe
                _bids = pd.DataFrame(data['bid'], index=data['dt'], columns=[symbol])
                _bids.index = pd.to_datetime(_bids.index)

                # transform data to volume dataframe
                _vols = pd.DataFrame(data['volume'], index=data['dt'], columns=[symbol])
                _vols.index = pd.to_datetime(_vols.index)

                # symbol dataframe to the daily portfolio
                asks = pd.merge(asks, _asks, how='outer', left_index=True, right_index=True)
                
                bids = pd.merge(bids, _bids, how='outer', left_index=True, right_index=True)

                vols = pd.merge(vols, _vols, how='outer', left_index=True, right_index=True)              

                # realese memory
                del data, _asks, _bids, _vols

            # ^ finished all symbols

            # fill nans with forward-fill (last non-nan price)
            asks.fillna(method='ffill', inplace=True)
            bids.fillna(method='ffill', inplace=True)

            # fill nans with backward-fill (for the first seconds of the week)
            asks.fillna(method='bfill', inplace=True)
            bids.fillna(method='bfill', inplace=True)

            # fill volume nans with 0
            vols.fillna(0, inplace=True)

            # create mid prices
            mids = ( asks + bids ) / 2
            
            # create spreads in %% pips
            spreads = ( asks / bids -1 ) * 10_000

            # create path ../data/primary/<year>/<week>/
            out_path = self.primary_path + str(year) +'/'+ str(wk) +'/'
            Path(out_path).mkdir(parents=True, exist_ok=True)

            # save daily csv into year week folder
            asks.to_csv(out_path + 'asks.csv', index=True)
            bids.to_csv(out_path + 'bids.csv', index=True)
            vols.to_csv(out_path + 'volumes.csv', index=True)
            mids.to_csv(out_path + 'mids.csv', index=True)
            spreads.to_csv(out_path + 'spreads.csv', index=True)

            print('\n...saving successfully asks.csv, bids.csv, mids.csv, spreads.csv, volumes.csv in', out_path, '\n')

            # realese memory
            del asks, bids, vols, mids, spreads

        # ^ finished all weeks

    ##_AsksBids.getAsksBids()


downData = PrimaryData()

downData.updateDB()







 