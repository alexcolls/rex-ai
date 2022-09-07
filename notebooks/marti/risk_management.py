import pandas as pd



RISK = 0.01
BALANCE = 100000
LEVERAGE = 1


class RiskManagement():
    def __init__(self):
        pass

    def getPrediction(self):
        pass


    def getLast(self):
        pass

    def correlation_pairs(self, df):

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


    def expected_volatility(self, pred_vol, correlations):

        vols = pred_vol.copy()
        pred_vol = abs(pred_vol)


        for sym in pred_vol.columns:

            real_vol = 0
            for sym2 in pred_vol.columns:

                if sym == sym2:
                    real_vol += pred_vol[sym]

                else:
                    real_vol += pred_vol[sym2] * correlations.filter(regex=sym+'_'+sym2).values[0][0]

            vols[sym] = abs(round(real_vol.mean(),6))

        total_vol = vols.mean(axis=1)[0]


        return total_vol, vols

    def mean_prediction(self, df, rate=24):
        data = pd.DataFrame([])
        for ccy in df.columns:
            data[ccy] = df[ccy].rolling(rate).mean()
        data.index = df.index

        return data

    def weighted_volatility(self, volatility):
        weight = pd.DataFrame([])
        volatility = volatility.copy()
        volatility_T = volatility.T
        weight = volatility_T/ volatility_T.mean()

        return  weight


    def read_exchange_rate(self, df):

        rate = df.iloc[-1:]

        data = pd.DataFrame(index=rate.index)

        for ccy in rate.columns:
            if ccy.split("_",2)[0] == "USD":
                data[(ccy.split("_")[1])] = rate[ccy]
            elif ccy.split("_",2)[1] == "USD":
                data[(ccy.split("_")[0])] = 1/rate[ccy]

        return data


    def trade_signals(self, pred_vols, last_data, classification, exchange_rate, risk=0.01, balance=100000, leverage=1):
        cor = correlation_pairs(last_data)

        t,v = expected_volatility(pred_vols, cor)
        n = int((abs(classification.T)).sum())
        risked = risk*balance*leverage/(t*n)


        trade_signals = []
        for ccy in classification.columns:
            d = {}
            if int(classification[ccy]) != 0:
                d["currency"] = ccy
                d["side"] = "sell" if int(classification[ccy]) == -1 else "buy"
                if ccy.split("_")[0] == "USD":
                    d["size"] = risked
                else:
                    d["size"] = risked*float(exchange_rate[ccy.split("_")[0]])
                d["datetime"] = v.index.to_list()[0].strftime("%Y-%m-%dT%H:%M:%S.Z")
                trade_signals.append(d)

        return trade_signals
