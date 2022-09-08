from risk_management import RiskManagement
from upload_gbq import upload_dataframe


def upload_trade_signals():

    rm = RiskManagement()

    print("\n### GETTING PREDICTIONS AND LAST DATA ###")
    pred_tend, pred_vol = rm.getPrediction()
    closes_df, logs_df = rm.getLast()

    last_data, pred_vol = rm.mean_volatility_prediction(logs_df)

    print("\n### ACCESING LAST EXANGING RATES ###")
    exchange_rate = rm.read_exchange_rate(closes_df)

    print("\n### SENDING BUY/SELL SIGNALS###")
    trade_signals, trade_signals_df = rm.trade_signals(
        pred_vol, last_data, pred_tend, exchange_rate
    )

    print("\n### BIG QUERY UPLOAD ###")
    upload_dataframe(exchange_rate, "exchange_rate")
    upload_dataframe(trade_signals_df, "trade_signals")


if __name__ == "__main__":
    upload_trade_signals()
