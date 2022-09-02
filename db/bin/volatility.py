import os
import pandas as pd
from db.bin.data_tertiary import TertiaryData
from indicators import bollinger_small, atr, time_standard
from pathlib import Path

class VolatilityFeatures(TertiaryData):
    def __init__(self):
        super().__init__()

    def getVolatility(self):


        DATA_PATH= os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../","data/merge")
        )

        print("Creating VOLATILITY Data Set")

        higs = pd.read_csv(os.path.join(DATA_PATH, "tertiary", "higs_.csv"), index_col=0)
        lows = pd.read_csv(os.path.join(DATA_PATH, "tertiary", "lows_.csv"), index_col=0)
        idxs = pd.read_csv(os.path.join(DATA_PATH, "tertiary", "idxs_.csv"), index_col=0)
        logs = pd.read_csv(os.path.join(DATA_PATH, "tertiary", "logs_.csv"), index_col=0)
        logs_= pd.read_csv(os.path.join(DATA_PATH, "secondary", "logs_.csv"), index_col=0)


        print("\n### BOLINGER BANDS ###")
        b = bollinger_small(df=logs, rate=48)
        # v = volatility(df=logs)
        print("\n### CALCULATING ATR ###")
        a = atr(idxs=idxs,low=lows,high=higs)
        # s = sharpe_ratio(df=logs)
        print("\n### TIME SCALING ###")
        time = time_standard(df=logs)

        data = b.join(a)
        # data = data.join(v)
        # data = data.join(s)
        data = logs_.join(data)
        data = data.join(time)

        file_path = os.path.join(DATA_PATH, "volatility")
        Path(file_path).mkdir(parents=True, exist_ok=True)
        data.to_csv(os.path.join(file_path, "volatility.csv"))

        return

if __name__ == "__main__":
    VolatilityFeatures().getVolatility()