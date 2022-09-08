# author: Quantium Rock & Roger Sole
# license: MIT

from datetime import datetime
from src.risk_manager import RiskManager


class Executor( RiskManager ):

    def __init__( self ):

        super(RiskManager, self).__init__()


    def sendOrders( self, new_orders: dict ):

        for sym, units in new_orders.items:
            try:
                self.postOrder(self.account_id, sym, units)
            except:
                print(f"Error in order: { sym, units } failed.")

        return True


if __name__ == "__main__":

    Executor()


# end