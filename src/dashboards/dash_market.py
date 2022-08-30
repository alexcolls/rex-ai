
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


dropDown = {}

db_path = 'db/data/primary/'
years = os.listdir(db_path)
for yr in years:
    try:
        dropDown[int(yr)] = []
        weeks = os.listdir(db_path+yr)
        for wk in weeks:
            try:
                dropDown[int(yr)].append(int(wk))
            except:
                continue
        dropDown[int(yr)].sort()
    except:
        continue        

current_year = list(dropDown.keys())[0]

# create the Dash app
app = dash.Dash('Model Dashboard')

params = [ 'Year', 'Week' ]

# set up the app layout
app.layout = html.Div(style={'margin': '80px'} ,children=[

    html.H1(children='MODEL DASHBOARD'),

    html.H3(children='Select year: '),

    dcc.Dropdown(id='year', options=[ {'label': yr, 'value': yr} for yr in dropDown.keys() ], value=current_year),

    html.H3(children='Select week: '),

    dcc.Dropdown(id='week', options=[{'label': wk, 'value': wk} for wk in dropDown[current_year] ], value=dropDown[current_year][-1]),

    html.H2(style={ 'margin-top': '50px'}, children='Market logarithmic returns (%)'),

        dcc.Graph(id='chart-mids'),

        dcc.Graph(id='chart-rets'),

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
    Output(component_id='chart-rets', component_property='figure'),
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
def selectWeek( year, week, filter_order=8, cutoff_freq=0.01  ):

    # plot mids returns
    mids_ = pd.read_csv('db/data/secondary/' + str(year) +'/'+ str(week) +'/'+ 'mids_.csv', index_col=0)
    mids_plt = px.line(mids_, height=900)

    # plot mids returns
    rets_ = pd.read_csv('db/data/secondary/' + str(year) +'/'+ str(week) +'/'+ 'rets_.csv', index_col=0)
    rets_plt = px.line(rets_, height=300)

    # plot spreads (%%) pips
    spreads_ = pd.read_csv('db/data/primary/' + str(year) +'/'+ str(week) +'/'+ 'spreads.csv', index_col=0)
    spreads_.index = pd.to_datetime(spreads_.index)
    spreads_plt = px.line(spreads_, height=300)

    # plot raw volumes
    volumes_ = pd.read_csv('db/data/primary/' + str(year) +'/'+ str(week) +'/'+ 'volumes.csv', index_col=0)
    volumes_plt = px.line(volumes_, height=300)
    
    # plot idxs returns
    idxs_ = pd.read_csv('db/data/secondary/' + str(year) +'/'+ str(week) +'/'+ 'idxs_.csv', index_col=0)

    lowpass_, idxs_lp = LowPass( idxs_, filter_order, cutoff_freq  )

    idxs_plt = px.line( idxs_lp, height=800)

    # trend
    trend_ = pd.DataFrame(index=idxs_.index, columns=idxs_.columns)

    for i in range(2, len(idxs_)):
        for ccy in idxs_.columns:
            trend_[ccy][i] = ( lowpass_[ccy][i] - lowpass_[ccy][i-1] )

    slope_threshold = 0.001

    trend_plt = px.line(trend_, height=400)
    trend_plt.add_hline(y=slope_threshold, line_color='green')
    trend_plt.add_hline(y=-slope_threshold, line_color='red')
    
    # substract singal noise from Low Pass filter

    noise_ = idxs_ - lowpass_

    noise_threshold = 0.1

    noise_plt = px.line(noise_, height=400)
    noise_plt.add_hline(y=noise_threshold, line_color='red')
    noise_plt.add_hline(y=-noise_threshold, line_color='green')    

    
    return mids_plt, rets_plt, spreads_plt, volumes_plt, idxs_plt, trend_plt, noise_plt



# run local server
if __name__ == '__main__':
    app.run_server(debug=True)