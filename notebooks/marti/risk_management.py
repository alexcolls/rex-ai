import pandas as pd
import numpy as np

def correlation_pairs(df, rate=120):

    data = pd.DataFrame([])
    for pair in df.columns:
        for second_pair in df.columns:
            data[f'{pair}_{second_pair}_corr'] = df[pair].rolling(120).corr(df[second_pair])
    data.index = df.index
    data = data.iloc[-1:].T
    data.columns = ["correlation"]
    data = data[round(data["correlation"],6) != 1.000000]
    return data

def expected_volatility(pred_vol, correlations):

    vols = pred_vol.copy().T
    correlations = correlations.copy().T

    for sym in pred_vol.columns:

        real_vol = 0

        for sym2 in pred_vol.columns:

            if sym == sym2:
                real_vol += pred_vol[sym]
            else:
                real_vol += pred_vol[sym2] * correlations.filter(regex=sym+'_'+sym2).values[0][0]

        vols[sym] = round( real_vol.mean(), 4)

    total_vol = vols.sum(axis=1)[0]

    return total_vol, vols

def weighted_volatility(volatility):
    weight = pd.DataFrame([])
    volatility = volatility.copy()
    volatility_T = volatility.T
    weight = volatility_T/ volatility_T.mean()
    return  weight
