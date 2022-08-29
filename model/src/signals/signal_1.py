
# author: Quantium Rock
# license: MIT

import pandas as pd
import plotly.express as px

from src.functions import LowPass
# LowPass filter params
filter_order = 8
cutoff_freq = 0.01

# Momentum threshold params
slope_threshold = 0.001

def Sigma_1( year=2021, week=2 ):

    mids_ = pd.read_csv('db/data/secondary/' + str(year) +'/'+ str(week) +'/'+ 'mids_.csv', index_col=0)

    # plot idxs returns
    idxs_ = pd.read_csv('db/data/secondary/' + str(year) +'/'+ str(week) +'/'+ 'idxs_.csv', index_col=0)

    lowpass_, idxs_lp = LowPass( idxs_, filter_order, cutoff_freq  )

    idxs_plt = px.line( idxs_lp, height=800)

    # trend
    trend_ = pd.DataFrame(index=idxs_.index, columns=idxs_.columns)

    for i in range(2, len(idxs_)):
        for ccy in idxs_.columns:
            trend_[ccy][i] = ( lowpass_[ccy][i] - lowpass_[ccy][i-1] )

    
    trend_plt = px.line(trend_, height=400)
    trend_plt.add_hline(y=slope_threshold, line_color='green')
    trend_plt.add_hline(y=-slope_threshold, line_color='red')
    
    idxs_plt.show() 
    trend_plt.show()


    timestamps = idxs_.index
    ccys = idxs_.columns
    thr = slope_threshold
    symbols = mids_.columns


    # create assets returns by timeframe

    rets_ = pd.DataFrame(index=timestamps, columns=symbols)

    for sym in symbols:
        last_tim = timestamps[0]
        for tim in timestamps:
            rets_[sym][tim] = mids_[sym][tim] - mids_[sym][last_tim]
            last_tim = tim

    #px.line(rets_).show()
            
    # currency indexes signals
    idxs_sigs = pd.DataFrame(index=idxs_.index, columns=idxs_.columns)

    for tim in timestamps:
        for ccy in ccys:
            if trend_[ccy][tim] > thr:
                idxs_sigs[ccy][tim] = 1
            elif trend_[ccy][tim] < -thr:
                idxs_sigs[ccy][tim] = -1
            else:
                idxs_sigs[ccy][tim] = 0

    px.line(idxs_sigs).show()

    assets_sigs = pd.DataFrame(index=mids_.index, columns=mids_.columns)

    for tim in timestamps:
        for sym in symbols:
            if idxs_sigs[sym[:3]][tim] > 0 and idxs_sigs[sym[4:]][tim] < 0:
                assets_sigs[sym][tim] = 1
            elif idxs_sigs[sym[:3]][tim] < 0 and idxs_sigs[sym[4:]][tim] > 0:
                assets_sigs[sym][tim] = -1
            else:
                assets_sigs[sym][tim] = 0

    px.line(assets_sigs).show()


    # backtest assets signals by asset

    sigmas = pd.DataFrame(index=timestamps, columns=symbols)

    for sym in symbols:
        for tim in timestamps:
            if assets_sigs[sym][tim] > 0:
                sigmas[sym][tim] = rets_[sym][tim]
            elif assets_sigs[sym][tim] < 0:
                sigmas[sym][tim] = -rets_[sym][tim]
            else:
                sigmas[sym][tim] = 0
                
    sigmas = sigmas.cumsum()

    # import spreads 
    spreads_ = pd.read_csv('db/data/primary/' + str(year) +'/'+ str(week) +'/'+ 'spreads.csv', index_col=0)

    brute_equity = sigmas.sum(axis=1) / len(symbols)

    #net_equity = brute_equity - spreads_ 
    #net_equity = net_equity.sum(axis=1) / len(symbols)



    px.line(sigmas).show()
    px.line(brute_equity).show()

    return 0


if __name__ == '__main__':
    Sigma_1(2021,2)
