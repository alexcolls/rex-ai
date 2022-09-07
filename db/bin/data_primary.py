# author: Quantium Rock
# license: MIT

import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd

from apis.oanda_api import OandaApi
from config import SYMBOLS, TIMEFRAME, START_YEAR


### PrimaryData -> download candles from oanda_api to ../data/primary/<year>/:
# opens.csv, highs.csv, lows.csv, closes.csv, volumes.csv

class PrimaryData:

    ## class constructor
    def __init__(self, symbols=SYMBOLS, timeframe=TIMEFRAME, start_year=START_YEAR):

        # quotes granularity default=5_second_bars
        self.symbols = symbols
        self.timeframe = timeframe
        self.start_year = start_year
        self.primary_path = os.path.normpath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "../", "data", "primary"
            )
        )
        self.db_path = self.primary_path
        Path(self.db_path).mkdir(parents=True, exist_ok=True)

    ## update data of primary db missing years & weeks
    def updateDB(self):

        start_time = datetime.utcnow()
        # if missing years download year
        if self.missing_years:
            for year in self.missing_years:
                print(year)
                self.getData(year=year)

        print("\nDB updated!")
        final_time = datetime.utcnow()
        duration = final_time - start_time
        print(
            "\nIt took",
            round(duration.total_seconds() / 60),
            "minutes to update the data.",
        )

        return True

    def deleteFolder(self, parent: str, year: str = "") -> None:
        import shutil

        if year:
            PATH_2022 = os.path.normpath(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "../",
                    "data",
                    parent,
                    year,
                )
            )
        else:
            PATH_2022 = os.path.normpath(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "../", "data", parent
                )
            )

        shutil.rmtree(PATH_2022, ignore_errors=True)
        print(
            f"Deleted {f'{parent}/{year}' if year else parent} directory to update from scratch"
        )
        return

    ## check missing years in db/data/
    def checkDB(self):

        # check missing years since <start_year>
        missing_years = []
        # current_year = datetime.utcnow().year
        years = [yr for yr in range(self.start_year, datetime.utcnow().year + 1)]

        # iterate each folder of db/data/primary/<year>/<week>/
        years_db = os.listdir(self.db_path)
        for year in years:
            if not str(year) in years_db:
                # append missing year
                print("Missing year:", year)
                missing_years.append(year)

        # if no asks_bids weeks missing
        if not missing_years:
            print("\nDB is fully updated since", self.start_year, "\n")

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

    def getData(self, year=2022):

        oanda_api = OandaApi()

        # current hour
        now = datetime.utcnow()

        day_of_year = now.timetuple().tm_yday - 1
        current_hour = now.hour

        first_date = datetime(year, 1, 2)
        first_date = first_date.replace(tzinfo=timezone.utc)

        year_ = now.year
        current_year = year == year_
        dtimes = []

        if not current_year:
            # get all trading hours of full year in datetime list
            for day_ in range(360):
                today = first_date + timedelta(days=day_)
                if today.weekday() < 5:
                    for hour_ in range(24):
                        dtimes.append(today + timedelta(hours=hour_))
        else:
            # get only datetimes until now
            for day_ in range(day_of_year):
                today = first_date + timedelta(days=day_)
                if today.weekday() < 5:
                    if day_ == day_of_year - 1:
                        for hour_ in range(current_hour + 1):
                            dtimes.append(today + timedelta(hours=hour_))
                    else:
                        for hour_ in range(24):
                            dtimes.append(today + timedelta(hours=hour_))

        # init daily dataframes indices for asks & bids
        op = pd.DataFrame(index=dtimes)
        hi = pd.DataFrame(index=dtimes)
        lo = pd.DataFrame(index=dtimes)
        cl = pd.DataFrame(index=dtimes)
        vo = pd.DataFrame(index=dtimes)

        # iterate each instrument to get a full week data each
        for symbol in self.symbols:

            start_date = str(year) + "-01-02T00:00:00.000000000Z"

            print(symbol)

            # initialize symbol data struct
            data = {
                "dtime": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
            }

            iterate = True
            # iterate until the end of year
            while iterate:

                # request 5000 bars from oanda rest-api
                req = oanda_api.getCandles(symbol, self.timeframe, start_date)

                # iterate each candle
                for x in req:
                    # if current candle time changed year
                    if pd.to_datetime(x["time"]).year > year:
                        iterate = False
                        break  # close year

                    # append data
                    data["dtime"].append(x["time"])
                    data["open"].append(float(x["mid"]["o"]))
                    data["high"].append(float(x["mid"]["h"]))
                    data["low"].append(float(x["mid"]["l"]))
                    data["close"].append(float(x["mid"]["c"]))
                    data["volume"].append(int(x["volume"]))

                    ## print(float(x['mid']['l']), x['time'])

                if len(data["dtime"]) > 0:
                    # only for current year: check if there is no more history
                    if start_date == data["dtime"][-1]:
                        iterate = False
                        del req
                        break

                    # otherwise update start_date with last loop request
                    start_date = data["dtime"][-1]

            # ^ finished symbol year

            # transform data to prices dataframe
            _op = pd.DataFrame(data["open"], index=data["dtime"], columns=[symbol])
            _op.index = pd.to_datetime(_op.index, utc=True)
            _hi = pd.DataFrame(data["high"], index=data["dtime"], columns=[symbol])
            _hi.index = pd.to_datetime(_hi.index, utc=True)
            _lo = pd.DataFrame(data["low"], index=data["dtime"], columns=[symbol])
            _lo.index = pd.to_datetime(_lo.index, utc=True)
            _cl = pd.DataFrame(data["close"], index=data["dtime"], columns=[symbol])
            _cl.index = pd.to_datetime(_cl.index, utc=True)
            _vo = pd.DataFrame(data["volume"], index=data["dtime"], columns=[symbol])
            _vo.index = pd.to_datetime(_vo.index, utc=True)

            op = pd.merge(op, _op, how="left", left_index=True, right_index=True)
            hi = pd.merge(hi, _hi, how="left", left_index=True, right_index=True)
            lo = pd.merge(lo, _lo, how="left", left_index=True, right_index=True)
            cl = pd.merge(cl, _cl, how="left", left_index=True, right_index=True)
            vo = pd.merge(vo, _vo, how="left", left_index=True, right_index=True)

            # realese memory
            del data, _op, _hi, _lo, _cl, _vo

        # ^ finished all symbols

        # fill nans with forward-fill (last non-nan price)
        op.fillna(method="ffill", inplace=True)
        hi.fillna(method="ffill", inplace=True)
        lo.fillna(method="ffill", inplace=True)
        cl.fillna(method="ffill", inplace=True)

        # fill nans with backward-fill (for the first seconds of the week)
        op.fillna(method="bfill", inplace=True)
        hi.fillna(method="bfill", inplace=True)
        lo.fillna(method="bfill", inplace=True)
        cl.fillna(method="bfill", inplace=True)

        # fill volume nans with 0
        vo.fillna(0, inplace=True)

        # create path ../data/primary/<year>/
        out_path = os.path.join(self.primary_path, str(year))
        Path(out_path).mkdir(parents=True, exist_ok=True)

        # save daily csv into year week folder
        op.to_csv(os.path.join(out_path, "opens.csv"), index=True)
        hi.to_csv(os.path.join(out_path, "highs.csv"), index=True)
        lo.to_csv(os.path.join(out_path, "lows.csv"), index=True)
        cl.to_csv(os.path.join(out_path, "closes.csv"), index=True)
        vo.to_csv(os.path.join(out_path, "volumes.csv"), index=True)

        print(
            "\n...saving successfully opens.csv, highs.csv, lows.csv, closes.csv, volumes.csv in",
            out_path,
            "\n",
        )

        # realese memory
        del op, hi, lo, cl, vo

        return True

    ##__primarData.getData()
