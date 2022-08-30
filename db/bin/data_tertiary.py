
# author: Quantium Rock
# license: MIT

import pandas as pd
from pathlib import Path
from data_secondary import SecondaryData


class TertiaryData ( SecondaryData ):


    def __init__ ( self ):

        super().__init__()
        self.tertiary_path = self.secondary_path.replace('secondary', 'tertiary')
        self.db_path = self.tertiary_path
        self.ccys = self.getCcys()        


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

    
    def getData ( self, year=2022 ):

        in_path = self.secondary_path + str(year) +'/'
        out_path = self.tertiary_path + str(year) +'/'

        Path(out_path).mkdir(parents=True, exist_ok=True)

        logs_ = pd.read_csv(in_path + 'logs_.csv', index_col=0)
        rets_ = pd.read_csv(in_path + 'rets_.csv', index_col=0)
        vols_ = pd.read_csv(in_path + 'vols_.csv', index_col=0)
        higs_ = pd.read_csv(in_path + 'higs_.csv', index_col=0)
        lows_ = pd.read_csv(in_path + 'lows_.csv', index_col=0)
        
        ln = len(self.ccys)

        for ccy in self.ccys:

            logs_base = logs_[logs_.filter(regex=ccy+'_').columns].sum(axis=1)
            logs_term = logs_[logs_.filter(regex='_'+ccy).columns].apply( lambda x: -x ).sum(axis=1)
            logs_[ccy] = ( logs_base + logs_term ) / ln

            rets_base = rets_[rets_.filter(regex=ccy+'_').columns].sum(axis=1)
            rets_term = rets_[rets_.filter(regex='_'+ccy).columns].apply( lambda x: -x ).sum(axis=1)
            rets_[ccy] = ( rets_base + rets_term ) / ln

            vols_[ccy] = vols_[vols_.filter(regex=ccy).columns].sum(axis=1) / ln
            higs_[ccy] = higs_[higs_.filter(regex=ccy).columns].sum(axis=1) / ln
            lows_[ccy] = lows_[lows_.filter(regex=ccy).columns].sum(axis=1) / ln

        logs_ = logs_[self.ccys]
        rets_ = rets_[self.ccys]
        vols_ = vols_[self.ccys]
        higs_ = higs_[self.ccys]
        lows_ = lows_[self.ccys]

        logs_.to_csv(out_path + 'logs_.csv', index=True)
        rets_.to_csv(out_path + 'rets_.csv', index=True)
        vols_.to_csv(out_path + 'vols_.csv', index=True)
        higs_.to_csv(out_path + 'higs_.csv', index=True)
        lows_.to_csv(out_path + 'lows_.csv', index=True)

        idxs_ = pd.DataFrame(index=rets_.index, columns=self.ccys)
        # create synthetic standarize idxs prices
        last_dt = 0
        for ccy in self.ccys:
            for i, dtime in enumerate(rets_.index):
                if i == 0:
                    idxs_[ccy][dtime] = 100
                    last_dt = dtime
                else:
                    idxs_[ccy][dtime] = idxs_[ccy][last_dt] * (1+rets_[ccy][dtime]/100)
                    last_dt = dtime

        idxs_.to_csv(out_path + 'idxs_.csv', index=True)

        del logs_, rets_, vols_, higs_, lows_, idxs_



        


