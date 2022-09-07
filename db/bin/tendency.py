import os

import pandas as pd
from data_tertiary import TertiaryData
from indicators import (
    rsi,
    ema,
    highpass_filter,
    lowpass_filter,
    time_standard,
    sharpe_ratio,
    lowpass_momentum,
    highpass_momentum,
    ema_diff,
    bollinger_small,
)
from pathlib import Path


class TendencyFeatures(TertiaryData):
    
    def __init__(self):
        super().__init__()

    def getTendency(self):

        DATA_PATH = os.path.normpath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "../", "data/merge"
            )
        )

        print("Creating TENDENCY Data Set")

        idxs = pd.read_csv(
            os.path.join(DATA_PATH, "tertiary", "idxs_.csv"), index_col=0
        )
        # idxs_= pd.read_csv(os.path.join(DATA_PATH, "secondary", "idxs_.csv"), index_col=0)
        time = time_standard(df=idxs)

        l = lowpass_filter(df=idxs)
        lm = lowpass_momentum(df=idxs)

        h = highpass_filter(df=idxs)
        hm = highpass_momentum(df=idxs)

        r8 = rsi(df=idxs, periods=8)
        r24 = rsi(df=idxs, periods=24)
        r120 = rsi(df=idxs, periods=120)

        e8 = ema(df=idxs, window=8)
        e24 = ema(df=idxs, window=24)
        e120 = ema(df=idxs, window=120)

        ed8 = ema_diff(df=idxs, window=8)
        ed24 = ema_diff(df=idxs, window=24)
        ed120 = ema_diff(df=idxs, window=120)

        s = sharpe_ratio(df=idxs, window=24)
        b = bollinger_small(df=idxs, rate=24)

        # f = fractals(df=idxs)

        data = l.join(h)
        data = data.join(lm)
        data = data.join(hm)
        data = data.join(r8)
        data = data.join(r24)
        data = data.join(r120)
        data = data.join(e8)
        data = data.join(e24)
        data = data.join(e120)
        data = data.join(s)
        data = data.join(b)
        data = data.join(ed8)
        data = data.join(ed24)
        data = data.join(ed120)
        # data = data.join(f)
        data = idxs.join(data)
        data = data.join(time, how="outer")

        file_path = os.path.join(DATA_PATH, "tendency")
        Path(file_path).mkdir(parents=True, exist_ok=True)
        data.to_csv(os.path.join(file_path, "tendency.csv"))

        return True


if __name__ == "__main__":
    TendencyFeatures().getTendency()
