import pandas as pd
import numpy as np

RISK = 0.01
BALANCE = 100000
LEVERAGE = 1



def correlation_pairs(df, rate=1):

    data = pd.DataFrame(index=df.index)
    for pair in df.columns:
        for second_pair in df.columns:
            data[f'{pair}_{second_pair}_corr'] = df[pair].corr(df[second_pair])
            # for the rolling correlation if we manage to do it...
            # data[f'{pair}_{second_pair}_corr'] = df[pair].rolling(rate).corr(df[second_pair])
    data.index = df.index
    data = data.iloc[-1:].T
    data.columns = ["correlation"]
    data = data[round(data["correlation"],6) != 1.000000]
    data = data.T

    return data


def expected_volatility(pred_vol, correlations):

    vols = pred_vol.copy()

    for sym in pred_vol.columns:

        real_vol = 0

        for sym2 in pred_vol.columns:

            if sym == sym2:
                real_vol += pred_vol[sym]
            else:
                real_vol += pred_vol[sym2] * correlations.filter(regex=sym+'_'+sym2).values[0][0]

        vols[sym] = round( real_vol.mean(), 4)

    total_vol = vols.mean(axis=1)[0]

    return total_vol, vols


def weighted_volatility(volatility):
    weight = pd.DataFrame([])
    volatility = volatility.copy()
    volatility_T = volatility.T
    weight = volatility_T/ volatility_T.mean()
    return  weight


def balance(pred_vols, classification, risk=1, balance=100000, leverage=1):
    cor = correlation_pairs(pred_vols)
    t,v = expected_volatility(pred_vols, cor)
    n = (abs(classification.T)).sum()
    risked = risk*balance*leverage/(t*n)
    print(risked)
    l = []
    for ccy in classification.columns:
        d = {}
        if int(classification[ccy]) != 0:
            d["ccy"] = ccy
            d["side"] = "sell" if int(classification[ccy]) == -1 else "buy"
            d["size"] = float(v[ccy].astype("float")*risked)
            d["datetime"] = pred_vols.index.to_list()[0].strftime("%Y-%m-%dT%H:%M:%S.Z")
            l.append(d)

    return l
