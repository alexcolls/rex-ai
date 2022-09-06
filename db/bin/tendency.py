import os
import pandas as pd
from data_tertiary import TertiaryData
from indicators import rsi, ema, highpass_filter, lowpass_filter, time_standard, sharpe_ratio
from pathlib import Path


class TendencyFeatures(TertiaryData):
    def __init__(self):
        super().__init__()

    def getTendency(self):

        DATA_PATH= os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../","data/merge")
        )

        print("Creating TENDENCY Data Set")

        idxs = pd.read_csv(os.path.join(DATA_PATH, "tertiary", "idxs_.csv"), index_col=0)
        idxs_= pd.read_csv(os.path.join(DATA_PATH, "secondary", "idxs_.csv"), index_col=0)
        time = time_standard(df=idxs)

        l = lowpass_filter(df=idxs)

        h = highpass_filter(df=idxs)

        r = rsi(df=idxs,periods=60)

        e = ema(df=idxs, window=14)

        s = sharpe_ratio(df=idxs, window=24)


        data = l.join(h)
        data = data.join(r)
        data = data.join(e)
        data = data.join(s)
        data = idxs_.join(data)
        data = data.join(time, how="outer")


        file_path = os.path.join(DATA_PATH, "tendency")
        Path(file_path).mkdir(parents=True, exist_ok=True)
        data.to_csv(os.path.join(file_path, "tendency.csv"))

        return True


if __name__ == "__main__":
    TendencyFeatures().getTendency()
