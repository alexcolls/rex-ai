
# author: Quantium Rock
# license: MIT

import numpy as np
import pandas as pd
from db.bin.secondary import SecondaryData



class TertiaryData ( SecondaryData ):

    def __init__ ( self ):
        super().__init__()
        self.tertiary_path = self.secondary_path.replace('secondary', 'tertiary')
        self.db_path = self.tertiary_path
