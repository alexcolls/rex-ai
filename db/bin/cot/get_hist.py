
# CONCAT FULL HISTORY OF G8 CURRENCIES COT POSITIONS BETWEEN SPECULATORS & COMMERCIALS

# author: Quantium Rock, MartiLlanes
# license: MIT
# date: October 2022

"""
    The Commodity Futures Trading Commission
    (US Derivatives Trading Commission or CFTC)
    publishes the Commitments of Traders (COT)
    reports to help the public understand market dynamics.
    Specifically, the COT reports provide a breakdown of each Tuesday’s open interest for futures and options
    on futures markets in which 20 or more traders hold positions equal to or above the reporting levels established by the CFTC.
"""

import pandas as pd

from pathlib import Path

from functools import reduce

import os



def net_positions(instrument):
    """ Computes the net position for large speculators and hedgers, where,

        net_position = long - short
    """
    large_speculators_net_position = instrument[2] - instrument[3]
    hedgers_net_position = instrument[5] - instrument[6]
    return large_speculators_net_position, hedgers_net_position

def market_sentiment(instrument):
    """ Compute market sentiment from Commitment of Traders report, where,

        instrument: is a list
        market sentiment = speculators net position - commercials net position
        market net position = long - short
    """
    non_commercials_net_position, commercials_net_position = net_positions(instrument)
    return non_commercials_net_position - commercials_net_position

def speculators_sentiment(instrument):
    """ Compute speculator position from CoT report, where,

        instrument: is a list
        speculators_long = # speculators long / (# spec. long + # spec. short)
        speculators_short = # speculators short / (# spec. long + # spec. short)
    """
    speculators_long = instrument[2] / (instrument[2] + instrument[3])
    speculators_short = instrument[3] / (instrument[2] + instrument[3])
    return round(speculators_long, 4), round(speculators_short, 4)


def compute_history():

    columns = [
        'Market and Exchange Names',
        'As of Date in Form YYYY-MM-DD',
        'Noncommercial Positions-Long (All)',
        'Noncommercial Positions-Short (All)',
        'Noncommercial Positions-Spreading (All)',
        'Commercial Positions-Long (All)',
        'Commercial Positions-Short (All)',
        'Nonreportable Positions-Long (All)',
        'Nonreportable Positions-Short (All)'
        ]

    contracts = {   'AUD': 'AUSTRALIAN DOLLAR',
                    'CAD': 'CANADIAN DOLLAR',
                    'CHF': 'SWISS FRA  NC',
                    'EUR': 'EURO FX',
                    'GBP': 'BRITISH POUND',
                    'JPY': 'JAPANESE YEN',
                    'NZD': 'NEW ZEALAND DOLLAR',
                    # USD = -(avg(of all ccys[:-2]))
            }

    speculators =   {   'AUD': [],
                        'CAD': [],
                        'CHF': [],
                        'EUR': [],
                        'GBP': [],
                        'JPY': [],
                        'NZD': [],
                    }

    # if you do headgers = speculators.copy() for some reason copyes the data all together
    hedgers = {   'AUD': [],
                        'CAD': [],
                        'CHF': [],
                        'EUR': [],
                        'GBP': [],
                        'JPY': [],
                        'NZD': [],
                    }


    for year in range(2022, 2004, -1):
        print(year)
        market_data = pd.read_csv(Path.cwd()/'cot'/ 'cots_raw'/ f'deacot{str(year)}' / 'annual.txt', delimiter=',', usecols=columns)
        "this is the direct implication since you call get hist from this folder"
        # market_data = pd.read_csv(Path.cwd() / cot_reports / 'deacot2018' / 'annual.txt', delimiter=',', usecols=columns)
        for ccy, cnts in contracts.items():
            for market in market_data.values:
                if cnts in market[0]:
                    market_delta = market_sentiment(market)
                    speculators_net_position, hedgers_net_position = net_positions(market)
                    speculators_long, speculators_short = speculators_sentiment(market)
                    speculators[ccy].append((market[1], speculators_net_position, speculators_long, speculators_short))
                    hedgers[ccy].append((market[1], hedgers_net_position))
                    print(f"week: {market[1]} (Tuesday)")
                    print(f'market sentiment: {market_delta}')
                    print(
                        'large speculators:',
                        f'net position = {speculators_net_position},',
                        f'long = {speculators_long} %,',
                        f'short = {speculators_short} % \n',
                    )
    return speculators, hedgers


def get_cot():
    speculators, hedgers = compute_history()
    dfs_s= []
    dfs_h = []
    for ccy in speculators.keys():
        if len(speculators[ccy]) != 0:
            dfs_s.append(pd.DataFrame(speculators[ccy]).rename(columns={0:"date",1:f"{ccy}_speculators_net_position",2:f"{ccy}_long",3:f"{ccy}_short"}))
        if len(hedgers[ccy]) != 0:
            dfs_h.append(pd.DataFrame(hedgers[ccy]).rename(columns={0:"date",1:f"{ccy}_hedgers_net_position"}))


    df_s = reduce(lambda  left,right: left.merge(right , left_on = "date" , right_on = "date", how = "outer"),dfs_s)
    df_h = reduce(lambda  left,right: pd.merge(left,right,on=['date'], how='outer'), dfs_h)
    df_s.fillna(0,inplace=True)
    df_h.fillna(0,inplace=True)

    data_cot_path = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath("cot.ipynb")), "../..", "data", "cot"
            )
        )
    os.makedirs(data_cot_path,exist_ok=True)
    df_s.to_csv(os.path.join(data_cot_path, "speculators.csv"), index=True)
    df_h.to_csv(os.path.join(data_cot_path, "hedgers.csv"), index=True)


    # del df_s, df_h, speculators,hedgers # Cleaning memory

    return True
