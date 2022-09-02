import os
import pandas as pd
from db.bin.data_tertiary import TertiaryData
from indicators import bollinger_small, atr, time_standard


class VolatilityFeatures(TertiaryData):
    def __init__(self):
        super().__init__()

    def getVolatility(self):


        DATA_PATH_TERTIARY= os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../", "data/merge/tertiary")
        )
        DATA_PATH_SECONDARY = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../", "data/merge/secondary")
        )
        print(DATA_PATH_TERTIARY) 
        print("Creating Volatility Data Set")

        higs = pd.read_csv(os.path.join(DATA_PATH_TERTIARY, "higs_.csv"), index_col=0)
        lows = pd.read_csv(os.path.join(DATA_PATH_TERTIARY, "lows_.csv"), index_col=0)
        idxs = pd.read_csv(os.path.join(DATA_PATH_TERTIARY, "idxs_.csv"), index_col=0)
        logs = pd.read_csv(os.path.join(DATA_PATH_TERTIARY, "logs_.csv"), index_col=0)
        logs_= pd.read_csv(os.path.join(DATA_PATH_SECONDARY, "logs_.csv"), index_col=0)
        # rets = pd.read_csv(os.path.join(DATA_PATH_TERTIARY, "rets_.csv"), index_col=0)

        print("Configuring Bollinger Bands...")
        bol = bollinger_small(df=logs, rate=48)
        # vol = volatility(df=logs)
        print("Calculating ATR...")
        at = atr(idxs=idxs,low=lows,high=higs)
        # sharpe = sharpe_ratio(df=logs)
        print("Time Scaling the Dates...")
        time = time_standard(df=logs)

        data = time.join(bol)
        data = bol.join(at)
        data = logs_.join(data)

        return data

if __name__ == "__main__":
    VolatilityFeatures().getVolatility()
