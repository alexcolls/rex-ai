
# author: Quantium Rock
# license: MIT

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from db.bin.primary import PrimaryData


class SecondaryData ( PrimaryData ):


    def __init__ ( self ):
        super().__init__()
        self.secondary_path = self.primary_path.replace('primary', 'secondary')
        self.db_path = self.secondary_path
        self.ccys = self.getCcys()


    def updateDB ( self ):

        start_time = datetime.utcnow()
        # if missing years download year
        if ( self.missing_years ):
            for year in self.missing_years:
                for week in range(1, 52):
                    try:
                        self.makeData( year=year, week=week )
                    except:
                        break
                                    
        # if missing weeks download weeks
        if ( self.missing_weeks ):
            for year, weeks in self.missing_weeks.items():
                for week in weeks:
                    self.makeData( year=year, week=week )
        
        print('\nDB updated!')

        final_time = datetime.utcnow()
        duration = final_time - start_time
        print('\nIt took', round(duration.total_seconds()/60/60, 2), 'hours to update the data.')

        return True


    def getCcys ( self ):
            ccys = []
            for sym in self.symbols:
                ccy = sym.split('_')
                if ccy[0] not in ccys:
                    ccys.append(ccy[0])
                if ccy[1] not in ccys:
                    ccys.append(ccy[1])
            ccys.sort()
            return ccys

    
    def makeData ( self, year, week ):

        in_path = self.primary_path + str(year) +'/'+ str(week) +'/'

        out_path = self.secondary_path + str(year) +'/'+ str(week) +'/'
        Path(out_path).mkdir(parents=True, exist_ok=True)

        # load mid prices
        mids = pd.read_csv(in_path + 'mids.csv', index_col=0)

        # create portfolio returns (standarize protfolio prices %)
        mids_ = ( np.log(mids) - np.log(mids.iloc[0]) )*100

        mids_.to_csv(out_path + 'mids_.csv', index=True)

        # create market logarithmic returns
        rets_ = ( np.log(mids) - np.log(mids.shift(1)) )*100

        rets_.to_csv(out_path + 'rets_.csv', index=True)

        # create currency indexes 
        idxs = pd.DataFrame(index=mids.index, columns=self.ccys)

        for ccy in self.ccys:
            base_ccys = mids[mids.filter(regex=ccy+'_').columns].apply( lambda x: 1/x ).sum(axis=1)
            term_ccys = mids[mids.filter(regex='_'+ccy).columns].sum(axis=1)
            idxs[ccy] = round( 1/ (( base_ccys + term_ccys + 1 ) / len(self.ccys)), 10)

        # standardize weekly currency indexes (%)

        idxs_ = ( np.log(idxs) - np.log(idxs.iloc[0]) )*100

        idxs_.to_csv(out_path + 'idxs_.csv', index=True)

        del mids, mids_, rets_, idxs, idxs_



        


