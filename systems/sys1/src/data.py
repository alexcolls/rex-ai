# author: Quantium Rock
# license: MIT

import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd
from apis.oanda_api import OandaApi
from systems.sys1.config import SYMBOLS, TIMEFRAME, START_YEAR


### PrimaryData -> download candles from oanda_api to ../data/primary/<year>/:
# opens.csv, highs.csv, lows.csv, closes.csv, volumes.csv

class PrimaryData:

    ## class constructor
    def __init__(self, symbols=SYMBOLS, timeframe=TIMEFRAME, start_year=START_YEAR):

        # quotes granularity
        self.symbols = symbols
        self.timeframe = timeframe
        self.start_year = start_year
        self.db_path = os.path.normpath(
            os.path.join( os.path.dirname(os.path.abspath(__file__)), "../data/series/" )
        )
        # create directories if doesn't exist
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

        return True

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


class SecondaryData(PrimaryData):

    def __init__( self ):

        super().__init__()
        Path(self.db_path).mkdir(parents=True, exist_ok=True)

    def getData(self, year=2022):

        in_path = os.path.join(self.primary_path, str(year))
        out_path = os.path.join(self.secondary_path, str(year))
        Path(out_path).mkdir(parents=True, exist_ok=True)

        # load primary data
        op = pd.read_csv(os.path.join(in_path, "opens.csv"), index_col=0)
        hi = pd.read_csv(os.path.join(in_path, "highs.csv"), index_col=0)
        lo = pd.read_csv(os.path.join(in_path, "lows.csv"), index_col=0)
        cl = pd.read_csv(os.path.join(in_path, "closes.csv"), index_col=0)

        # create portfolio returns (standarize protfolio prices %)
        logs_ = (np.log(cl) - np.log(op)) * 100
        rets_ = (cl / op - 1) * 100
        vols_ = (hi / lo - 1) * 100
        higs_ = (hi / cl - 1) * 100
        lows_ = (cl / lo - 1) * 100

        logs_.to_csv(os.path.join(out_path, "logs_.csv"), index=True)
        rets_.to_csv(os.path.join(out_path, "rets_.csv"), index=True)
        vols_.to_csv(os.path.join(out_path, "vols_.csv"), index=True)
        higs_.to_csv(os.path.join(out_path, "higs_.csv"), index=True)
        lows_.to_csv(os.path.join(out_path, "lows_.csv"), index=True)

        del op, hi, lo, cl, logs_, rets_, vols_, higs_, lows_

        return True




class TertiaryData(SecondaryData):

    def __init__(self):

        super().__init__()
        self.tertiary_path = self.secondary_path.replace("secondary", "tertiary")
        self.db_path = self.tertiary_path
        Path(self.db_path).mkdir(parents=True, exist_ok=True)
        self.ccys = self.getCcys()

    def getCcys(self):
        ccys = []
        for sym in self.symbols:
            ccy = sym.split("_")
            if ccy[0] not in ccys:
                ccys.append(ccy[0])
            if ccy[1] not in ccys:
                ccys.append(ccy[1])
        ccys.sort()
        return ccys

    def getData(self, year=2022):

        in_path = os.path.join(self.secondary_path, str(year))
        out_path = os.path.join(self.tertiary_path, str(year))

        Path(out_path).mkdir(parents=True, exist_ok=True)

        logs_ = pd.read_csv(os.path.join(in_path, "logs_.csv"), index_col=0)
        rets_ = pd.read_csv(os.path.join(in_path, "rets_.csv"), index_col=0)
        vols_ = pd.read_csv(os.path.join(in_path, "vols_.csv"), index_col=0)
        higs_ = pd.read_csv(os.path.join(in_path, "higs_.csv"), index_col=0)
        lows_ = pd.read_csv(os.path.join(in_path, "lows_.csv"), index_col=0)

        ln = len(self.ccys)

        for ccy in self.ccys:

            logs_base = logs_[logs_.filter(regex=ccy + "_").columns].sum(axis=1)
            logs_term = (
                logs_[logs_.filter(regex="_" + ccy).columns]
                .apply(lambda x: -x)
                .sum(axis=1)
            )
            logs_[ccy] = (logs_base + logs_term) / ln

            rets_base = rets_[rets_.filter(regex=ccy + "_").columns].sum(axis=1)
            rets_term = (
                rets_[rets_.filter(regex="_" + ccy).columns]
                .apply(lambda x: -x)
                .sum(axis=1)
            )
            rets_[ccy] = (rets_base + rets_term) / ln

            vols_[ccy] = vols_[vols_.filter(regex=ccy).columns].sum(axis=1) / ln
            higs_[ccy] = higs_[higs_.filter(regex=ccy).columns].sum(axis=1) / ln
            lows_[ccy] = lows_[lows_.filter(regex=ccy).columns].sum(axis=1) / ln

        logs_ = logs_[self.ccys]
        rets_ = rets_[self.ccys]
        vols_ = vols_[self.ccys]
        higs_ = higs_[self.ccys]
        lows_ = lows_[self.ccys]

        logs_.to_csv(os.path.join(out_path, "logs_.csv"), index=True)
        rets_.to_csv(os.path.join(out_path, "rets_.csv"), index=True)
        vols_.to_csv(os.path.join(out_path, "vols_.csv"), index=True)
        higs_.to_csv(os.path.join(out_path, "higs_.csv"), index=True)
        lows_.to_csv(os.path.join(out_path, "lows_.csv"), index=True)

        idxs_ = pd.DataFrame(index=rets_.index, columns=self.ccys)
        # create synthetic standarize idxs prices
        last_dt = 0
        for ccy in self.ccys:
            for i, dtime in enumerate(rets_.index):
                if i == 0:
                    idxs_[ccy][dtime] = 100
                    last_dt = dtime
                else:
                    idxs_[ccy][dtime] = idxs_[ccy][last_dt] * (
                        1 + rets_[ccy][dtime] / 100
                    )
                    last_dt = dtime

        idxs_.to_csv(os.path.join(out_path, "idxs_.csv"), index=True)

        del logs_, rets_, vols_, higs_, lows_, idxs_



# main for function call.
if __name__ == "__main__":

