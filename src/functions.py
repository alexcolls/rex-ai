

import scipy.signal as signal

def LowPass( df, filter_order=8, cutoff_freq=1.0 ):

    # buterworth filter
    B, A = signal.butter(filter_order, cutoff_freq, output='ba')
    
    df1 = df.copy()
    df2 = df1.copy()

    # apply lowpass filter
    for ccy in df.columns:
        df1[ccy] = signal.filtfilt(B, A, df[ccy])
        df2[ccy+'_lp'] = df1[ccy]
    
    return df1, df2


