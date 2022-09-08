import pandas as pd
from datetime import datetime, timezone
from apis.oanda_api import OandaApi
from config import ACCOUNT


def create_orders(df: pd.DataFrame, units: int = 1000) -> None:

    oapi = OandaApi()

    orders = df.loc[:, df.values[0] != 0]

    for order, value in zip(orders, orders.values[0]):
        units = value * units
        try:
            oapi.postOrder(ACCOUNT, order, units)
        except:
            print(f"Error in order: {ACCOUNT, order, units}")

    return


if __name__ == "__main__":
    data = {
        "AUD_CAD": 1,
        "AUD_CHF": -1,
        "AUD_JPY": -1,
        "AUD_NZD": 1,
        "AUD_USD": 0,
        "CAD_CHF": 1,
        "CAD_JPY": 0,
        "CHF_JPY": -1,
        "EUR_AUD": 1,
        "EUR_CAD": -1,
        "EUR_CHF": 0,
        "EUR_GBP": -1,
        "EUR_JPY": 1,
        "EUR_NZD": 0,
        "EUR_USD": 0,
        "GBP_AUD": -1,
        "GBP_CAD": 0,
        "GBP_CHF": 1,
        "GBP_JPY": -1,
        "GBP_NZD": 0,
        "GBP_USD": 1,
        "NZD_CAD": -1,
        "NZD_CHF": 0,
        "NZD_JPY": 1,
        "NZD_USD": -1,
        "USD_CAD": 0,
        "USD_CHF": 0,
        "USD_JPY": 1,
    }

    df = pd.DataFrame(data, index=[datetime(2022, 9, 8, 0, 0, 0, 0, timezone.utc)])
    create_orders(df)
