"""MC2-P1: Market simulator.

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---

Student Name: Sashank Gondala (replace with your name)
GT User ID: sgondala3 (replace with your User ID)
GT ID: 903388899 (replace with your GT ID)
"""

import pandas as pd
import numpy as np
import datetime as dt
import os
from indicators import CONSTANTS

from util import get_data

def author():
    return 'sgondala3'

# Buying at the end of
def compute_portvals(pd_orders, symbol = "JPM", start_val = 1000000,
                     commission=9.95, impact=0.005):

    assert isinstance(pd_orders, pd.DataFrame)

    pd_orders.sort_index(inplace=True)

    all_symbols = [symbol]
    bSPYPresent = True if CONSTANTS.SPY in all_symbols else False

    dates = pd_orders.index.values
    start_date = dates[0]
    end_date = dates[-1]

    # Reading prices and filling in missing values
    prices = get_data(all_symbols, pd.date_range(start_date, end_date))
    prices.ffill(inplace=True)
    prices.bfill(inplace=True)
    if not bSPYPresent:
        prices.drop(CONSTANTS.SPY, axis=1, inplace=True)

    buy_sell_df = prices.copy()
    buy_sell_df[CONSTANTS.CASH] = 0
    buy_sell_df.ix[:,:] = 0

    for date, row in pd_orders.iterrows():
        # Orders placed on non trading days are ignored
        if date not in prices.index:
            continue
        date_traded = date
        symbol_traded = symbol
        number_traded = row[CONSTANTS.TRADES]

        price_executed = prices.ix[date_traded, symbol_traded]
        if number_traded > 0:
            price_executed += impact*price_executed
        else:
            price_executed -= impact*price_executed

        cash_traded = number_traded*price_executed

        buy_sell_df.ix[date_traded, symbol_traded] += number_traded
        buy_sell_df.ix[date_traded, CONSTANTS.CASH] -= cash_traded
        buy_sell_df.ix[date_traded, CONSTANTS.CASH] -= commission

    cumm_df = buy_sell_df.cumsum()
    cumm_df[CONSTANTS.CASH] += start_val

    portvals = pd.DataFrame(index = cumm_df.index)
    portvals[CONSTANTS.TOTAL] = 0

    columns = cumm_df.columns

    for date, row in cumm_df.iterrows():
        for column in columns:
            if column == CONSTANTS.CASH:
                portvals.ix[date, CONSTANTS.TOTAL] += row[column]
            else:
                portvals.ix[date, CONSTANTS.TOTAL] += row[column] * prices.ix[date, column]

    rv = pd.DataFrame(index=portvals.index, data=portvals.as_matrix())

    # return rv
    return portvals

# def test_code():
#     # this is a helper function you can use to test your code
#     # note that during autograding his function will not be called.
#     # Define input parameters
#
#     of = "./orders/orders2.csv"
#     sv = 1000000
#
#     # Process orders
#     portvals = compute_portvals(orders_file = of, start_val = sv)
#     if isinstance(portvals, pd.DataFrame):
#         portvals = portvals[portvals.columns[0]] # just get the first column
#     else:
#         "warning, code did not return a DataFrame"
#
#     # Get portfolio stats
#     # Here we just fake the data. you should use your code from previous assignments.
#     start_date = portvals.index.values[0]
#     end_date = portvals.index.values[-1]
#     # cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = \
#     #    get_cumm_returns(portvals), get_avg_daily_returns(portvals), get_std_daily_returns(portvals), get_sharpe_ratio(portvals)
#     cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]
#     cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]
#
#     # Compare portfolio against $SPX
#     print "Date Range: {} to {}".format(start_date, end_date)
#     print
#     print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
#     print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
#     print
#     print "Cumulative Return of Fund: {}".format(cum_ret)
#     print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
#     print
#     print "Standard Deviation of Fund: {}".format(std_daily_ret)
#     print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
#     print
#     print "Average Daily Return of Fund: {}".format(avg_daily_ret)
#     print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
#     print
#     print "Final Portfolio Value: {}".format(portvals[-1])

# 1000 - Bought 1000
# -2000 - Sold 2000
if __name__ == "__main__":
    data = {'Trades':[1000, -2000, 1000]}
    df_trades = pd.DataFrame(data, index=[dt.datetime(2010, 1, 4), dt.datetime(2010, 1, 5), dt.datetime(2010, 1, 6)])
    compute_portvals(df_trades)