
# author: Quantium Rock
# license: MIT

import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import os

from src.functions import LowPass
# LowPass filter params
filter_order = 8
cutoff_freq = 0.01

# Momentum threshold params
slope_threshold = 0.001

def dropDown():
    drop_down = {}

    db_path = 'db/data/primary/'
    years = os.listdir(db_path)
    for yr in years:
        try:
            drop_down[int(yr)] = []
            weeks = os.listdir(db_path+yr)
            for wk in weeks:
                try:
                    drop_down[int(yr)].append(int(wk))
                except:
                    continue
            drop_down[int(yr)].sort()
        except:
            continue        

    current_year = list(dropDown.keys())[0]

    return drop_down, current_year

dop_down, current_year = dropDown()

# create the Dash app
app = dash.Dash('Model Dashboard')

params = [ 'Year', 'Week' ]

# set up the app layout
app.layout = html.Div(style={'margin': '80px'}, children=[

    html.H1(children='MODEL DASHBOARD'),

    html.H3(children='Select year: '),

    dcc.Dropdown(id='year', options=[ {'label': yr, 'value': yr} for yr in dop_down.keys() ], value=current_year),

    html.H3(children='Select week: '),

    dcc.Dropdown(id='week', options=[{'label': wk, 'value': wk} for wk in dop_down[current_year] ], value=dop_down[current_year][-1]),

    html.H2(style={ 'margin-top': '50px'}, children='Market logarithmic returns (%)'),

        dcc.Graph(id='chart-mids'),

        html.H4(children='Spreads in pips (%%)'),

        dcc.Graph(id='chart-spreads'),

        html.H4(children='Raw Traded Volumes'),

        dcc.Graph(id='chart-volumes'),

    html.H2(children='Market dimensional reduction to Currency Indexes (%)'),

        html.H4(children='Currency logarithmic returns + LowPass Filter (filter_order=8, cutoff_freq=0.01)'),

        dcc.Graph(id='chart-idxs'),

        #dcc.Slider(id='filter_order'),
        #dcc.Slider(id='cutoff_freq'),

        html.H4(children='Low Frequency Momentum (LP slope = log(LP[0]) - log(LP[1]) )'),

        dcc.Graph(id='chart-trend'),

        html.H4(children='High Frequency Noise (LP deviations = log(value[0]) - log(LP[0]) )'),

        dcc.Graph(id='chart-noise'),
        
])

# set up callback function
@ app.callback(

    Output(component_id='chart-mids', component_property='figure'),
    Output(component_id='chart-spreads', component_property='figure'),
    Output(component_id='chart-volumes', component_property='figure'),

    Output(component_id='chart-idxs', component_property='figure'),
    Output(component_id='chart-trend', component_property='figure'),
    Output(component_id='chart-noise', component_property='figure'),

    Input(component_id='year', component_property='value'),
    Input(component_id='week', component_property='value'),

    #Input(component_id='filter_order', component_property='value'),
    #Input(component_id='cutoff_freq', component_property='value')

)
def selectWeek( year, week ):

    mids_ = pd.read_csv('db/data/secondary/' + str(year) +'/'+ str(week) +'/'+ 'mids_.csv', index_col=0)

    mids_plt = px.line(mids_)

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
    

    # prepare signal 

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

    assets_sigs = px.line(assets_sigs).show()

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
    #spreads_ = pd.read_csv('db/data/primary/' + str(year) +'/'+ str(week) +'/'+ 'spreads.csv', index_col=0)

    equity = sigmas.sum(axis=1) / len(symbols)

    #net_equity = brute_equity - spreads_ 
    #net_equity = net_equity.sum(axis=1) / len(symbols)

    px.line(sigmas).show()
    px.line(equity).show()

    return 0


# run local server
if __name__ == '__main__':
    app.run_server(debug=True)