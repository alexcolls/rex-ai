# author: Quantium Rock & Roger Sole
# license: MIT

import pandas as pd
from datetime import datetime, timezone
from libs.oanda_api import OandaApi
from risk_manager import RiskManager


class Executor( RiskManager ):

    def __init__( self ):

        super().__init__()


    def create_orders( self, df: pd.DataFrame, units: int=1000) -> None:

        oapi = OandaApi()

        orders = df.loc[:, df.values[0] != 0]

        for order, value in zip(orders, orders.values[0]):
            units = value * units
            try:
                oapi.postOrder(self.account, order, units)
            except:
                print(f"Error in order: {self.account, order, units}")

        return True


if __name__ == "__main__":

    pass
