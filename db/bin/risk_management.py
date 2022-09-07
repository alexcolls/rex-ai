import pandas as pd
from gbq_utils import load_last_rows


RISK = 0.01
BALANCE = 100000
LEVERAGE = 1


class RiskManagement():
    def __init__(self):
        pass

    def getPrediction(self):
        tend_prediction_df = load_last_rows("tendency_pred",1)
        vol_prediction_df = load_last_rows("volatility_pred",1)
        return tend_prediction_df, vol_prediction_df



    def getLast(self):
        primary_df = load_last_rows("closes",1)
        tertiary_logs_df = load_last_rows("logs_",240)
        return primary_df, tertiary_logs_df

    def mean_volatility_prediction(self, logs, rate=120):
        data = pd.DataFrame([])
        for ccy in logs.columns:
            data[ccy] = logs[ccy].rolling(rate).mean() + logs[ccy].rolling(rate).std()*2
        data.index = logs.index
        data2 = data.iloc[-1:].copy()

        return data, data2

    def correlation_pairs(self, pred_vols):

        data = pd.DataFrame(index=pred_vols.index)
        for pair in pred_vols.columns:
            for second_pair in pred_vols.columns:
                data[f'{pair}_{second_pair}_corr'] = pred_vols[pair].corr(pred_vols[second_pair])

        data.index = pred_vols.index
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


    def trade_signals(self,pred_vols,last_data, classification, exchange_rate, risk=0.01, balance=100000, leverage=1):
        cor = self.correlation_pairs(last_data)

        t,v = self.expected_volatility(pred_vols, cor)
        n = int((abs(classification.T)).sum())
        risked = risk*balance*leverage/(t*n)


        trade_signals = []
        trade_dataframe = {"datetime":[],"currency":[], "side":[], "size":[]}
        for ccy in classification.columns:
            d = {}
            if int(classification[ccy]) != 0:
                d["currency"] = ccy
                trade_dataframe["currency"].append(ccy)
                signal = "sell" if int(classification[ccy]) == -1 else "buy"
                d["side"] = signal
                trade_dataframe["side"].append(signal)
                if ccy.split("_")[0] == "USD":
                    d["size"] = round(risked)
                    trade_dataframe["size"].append(round(risked))
                else:
                    d["size"] = round(risked*float(exchange_rate[ccy.split("_")[0]]))
                    trade_dataframe["size"].append(round(risked*float(exchange_rate[ccy.split("_")[0]])))
                d["datetime"] = v.index.to_list()[0].strftime("%Y-%m-%dT%H:%M:%S.Z")
                trade_dataframe["datetime"].append(v.index.to_list()[0])

                trade_signals.append(d)
        trade_dataframe = pd.DataFrame.from_dict(trade_dataframe)
        trade_dataframe.set_index("datetime",inplace=True)
        trade_dataframe.index = pd.to_datetime(trade_dataframe.index)

        return trade_signals, trade_dataframe
