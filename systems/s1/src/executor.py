# author: Quantium Rock & Roger Sole
# license: MIT

import time
from datetime import datetime
from src.risk_manager import RiskManager


class Executor( RiskManager ):

    def __init__( self ):

        super().__init__()

        print(self.new_orders)


    def sendOrders( self, new_orders: dict ):

        for sym, units in new_orders.items():
            try:
                self.postOrder(self.account_id, sym, units)
            except:
                print(f"Error in order: { sym, units } failed.")

        return True

    
    def closeAll( self ):

        positions = self.getOpenPositions(self.account_id)

        for pos in positions:

            sym = pos['instrument']
            long = int(pos['long']['units'])
            short = int(pos['short']['units'])

            if long:
                print(self.postOrder(self.account_id, sym, -long))
            elif short:
                print(self.postOrder(self.account_id, sym, -short))



if __name__ == "__main__":

    ex = Executor()

    ex.closeAll()

    #time.sleep(10)
    #ex.sendOrders(ex.new_orders)


# end