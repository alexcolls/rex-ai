
import time
from datetime import datetime
from src.executor import Executor


class Main( Executor ):

    def __init__( self ):

        super().__init__()


    def trade( self ):
        self.sendOrders(self.new_orders)

    def appendData( self ):
        pass

    def uploadGBQ( self ):
        pass



if __name__ == '__main__':

    algo = Main()

    while True:
        time.sleep(60)
        timestamp = datetime.utcnow()
        print(timestamp)
        algo.trade()