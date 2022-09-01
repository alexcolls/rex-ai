
# author: Quantium Rock
# license: MIT

import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import os

drop_years = []
path = 'db/data/primary/'

years = os.listdir(path)
for yr in years:
    try:
        drop_years.append(int(yr))
    except:
        continue    

# create the Dash app
app = dash.Dash()

# set up the app layout
app.layout = html.Div(style={'margin': '80px'} ,children=[

    html.H1(children='DB DASHBOARD'),

    html.H3(children='Select year: '),

    dcc.Dropdown(id='year', options=[ {'label': yr, 'value': yr} for yr in drop_years ], value=2021),

    html.H3(children='Select database: '),

    dcc.Dropdown(id='db_level', options=[ 'secondary', 'tertiary' ], value='tertiary'),

    html.H2(children='___'),

    html.H4(children='Cumulative Log Returns (%)'),

    dcc.Graph(id='chart-idxs'),

    html.H4(children='Logarithmic Returns (%)'),

    dcc.Graph(id='chart-logs'),

    html.H4(children='Returns (%)'),

    dcc.Graph(id='chart-rets'),

    html.H4(children='Volatilities (%)'),

    dcc.Graph(id='chart-vols'),

    html.H4(children='Highs (%)'),

    dcc.Graph(id='chart-higs'),

    html.H4(children='Lows (%)'),

    dcc.Graph(id='chart-lows'),
        
])

# set up callback function
@ app.callback(

    Output(component_id='chart-idxs', component_property='figure'),
    Output(component_id='chart-logs', component_property='figure'),
    Output(component_id='chart-rets', component_property='figure'),

    Output(component_id='chart-vols', component_property='figure'),
    Output(component_id='chart-higs', component_property='figure'),
    Output(component_id='chart-lows', component_property='figure'),

    Input(component_id='year', component_property='value'),
    Input(component_id='db_level', component_property='value'),

)
def selectWeek( year, db_level ):

    db_path = f'db/data/{db_level}/' 

    # load tertiary db
    idxs_ = pd.read_csv(db_path + str(year) + '/idxs_.csv', index_col=0)
    logs_ = pd.read_csv(db_path + str(year) + '/logs_.csv', index_col=0)
    rets_ = pd.read_csv(db_path + str(year) + '/rets_.csv', index_col=0)
    vols_ = pd.read_csv(db_path + str(year) + '/vols_.csv', index_col=0)
    higs_ = pd.read_csv(db_path + str(year) + '/higs_.csv', index_col=0)
    lows_ = pd.read_csv(db_path + str(year) + '/lows_.csv', index_col=0)

    idxs_.reset_index(drop=True, inplace=True)
    idxs_plt = px.line(idxs_, height=600)
    logs_.reset_index(drop=True, inplace=True)
    logs_plt = px.line(logs_, height=400)
    rets_.reset_index(drop=True, inplace=True)
    rets_plt = px.line(rets_, height=400)
    vols_.reset_index(drop=True, inplace=True)
    vols_plt = px.line(vols_, height=400)
    higs_.reset_index(drop=True, inplace=True)
    higs_plt = px.line(higs_, height=400)
    lows_.reset_index(drop=True, inplace=True)
    lows_plt = px.line(lows_, height=400)
    
    return idxs_plt, logs_plt, rets_plt, vols_plt, higs_plt, lows_plt


# run local server
if __name__ == '__main__':
    app.run_server(debug=True)