from curses import window
import os
import math
from re import S
import pandas as pd
import numpy as np
import scipy.signal as sig
from db.bin.data_secondary import SecondaryData
from db.bin.data_tertiary import TertiaryData
from indicators import rsi, ema, highpass_filter, lowpass_filter, time_standard


class TendencyFeatures(TertiaryData):
    def __init__(self):
        super().__init__()

    def getVolatility(self):

        DATA_PATH = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../", "data/merge/tertiary")
        )

        higs = pd.read_csv(os.path.join(DATA_PATH, "high_.csv"), index_col=0)
        lows = pd.read_csv(os.path.join(DATA_PATH, "lows_.csv"), index_col=0)
        # idxs = pd.read_csv(os.path.join(DATA_PATH, "idxs_.csv"), index_col=0)
        logs = pd.read_csv(os.path.join(DATA_PATH, "logs_.csv"), index_col=0)
        # rets = pd.read_csv(os.path.join(DATA_PATH, "rets_.csv"), index_col=0)

        lowpass = lowpass_filter(df=logs)
        highpass = highpass_filter(df=logs)
        rsi = rsi(df=logs,window=240)
        ema = ema(df=logs, window=48)
