import os
import pandas as pd
from db.bin.data_tertiary import TertiaryData
from indicators import rsi, ema, highpass_filter, lowpass_filter, time_standard
from pathlib import Path


class TendencyFeatures(TertiaryData):
    def __init__(self):
        super().__init__()

    def getTendency(self):

        DATA_PATH= os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../","data/merge")
        )

        print("Creating TENDENCY Data Set")

        logs = pd.read_csv(os.path.join(DATA_PATH, "tertiary", "logs_.csv"), index_col=0)
        logs_= pd.read_csv(os.path.join(DATA_PATH, "secondary", "logs_.csv"), index_col=0)
        time = time_standard(df=logs)
        print("\n### LOWPASS FILTERING ###")
        l = lowpass_filter(df=logs)
        print("\n### HIGHPASS FILTERING ###")
        h = highpass_filter(df=logs)
        print("\n### CALCULATING RSI ###")
        r = rsi(df=logs,periods=240)
        print("\n### CALCULATING EMA ###")
        e = ema(df=logs, window=48)
        print("\n### TIME SCALING ###")


        data = l.join(h)
        data = data.join(r)
        data = data.join(e)
        data = logs_.join(data)
        data = data.join(time, how="outer")

        print("\n### CREATING tendency.csv ###")
        file_path = os.path.join(DATA_PATH, "tendency")
        Path(file_path).mkdir(parents=True, exist_ok=True)
        data.to_csv(os.path.join(file_path, "tendency.csv"))

        return


if __name__ == "__main__":
    TendencyFeatures().getTendency()
