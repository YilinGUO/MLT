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

from util import get_data, plot_data

def author():
    return 'sgondala3'

def compute_portvals(orders_file = "./orders/orders-01.csv", start_val = 1000000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code 			  		 			     			  	   		   	  			  	
    # NOTE: orders_file may be a string, or it may be a file object. Your 			  		 			     			  	   		   	  			  	
    # code should work correctly with either input

    pd_orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    pd_orders.sort_index(inplace=True)

    all_symbols = pd_orders['Symbol'].unique().tolist()
    bSPYPresent = True if 'SPY' in all_symbols else False

    dates = pd_orders.index.values
    start_date = dates[0]
    end_date = dates[-1]

    # Reading prices and filling in missing values
    prices = get_data(all_symbols, pd.date_range(start_date, end_date))
    prices.ffill(inplace=True)
    prices.bfill(inplace=True)
    if not bSPYPresent:
        prices.drop('SPY', axis=1, inplace=True)

    buy_sell_df = prices.copy()
    buy_sell_df['Cash'] = 0
    buy_sell_df.ix[:,:] = 0

    for date, row in pd_orders.iterrows():
        # Orders placed on non trading days are ignored
        if date not in prices.index:
            continue
        date_traded = date
        symbol_traded = row['Symbol']
        type_of_trade = row['Order']
        number_traded = row['Shares']
        if type_of_trade == 'SELL':
            number_traded = -number_traded

        price_executed = prices.ix[date_traded, symbol_traded]
        if type_of_trade == 'SELL':
            price_executed -= impact*price_executed
        else:
            price_executed += impact*price_executed

        cash_traded = number_traded*price_executed

        buy_sell_df.ix[date_traded, symbol_traded] += number_traded
        buy_sell_df.ix[date_traded, 'Cash'] -= cash_traded
        buy_sell_df.ix[date_traded, 'Cash'] -= commission

    cumm_df = buy_sell_df.cumsum()
    cumm_df['Cash'] += start_val

    portvals = pd.DataFrame(index = cumm_df.index)
    portvals['Total'] = 0

    columns = cumm_df.columns

    for date, row in cumm_df.iterrows():
        for column in columns:
            if column == 'Cash':
                portvals.ix[date, 'Total'] += row[column]
            else:
                portvals.ix[date, 'Total'] += row[column] * prices.ix[date, column]

    rv = pd.DataFrame(index=portvals.index, data=portvals.as_matrix())

    # return rv
    return portvals

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders2.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = portvals.index.values[0]
    end_date = portvals.index.values[-1]
    # cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = \
    #    get_cumm_returns(portvals), get_avg_daily_returns(portvals), get_std_daily_returns(portvals), get_sharpe_ratio(portvals)
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])
 			  		 			     			  	   		   	  			  	
if __name__ == "__main__": 			  		 			     			  	   		   	  			  	
    compute_portvals()
