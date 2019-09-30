"""MC1-P2: Optimize a portfolio. 			  		 			     			  	   		   	  			  	
																				
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
from util import get_data, plot_data
from scipy.optimize import minimize	  	
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_portfolio_values(df, alloc, initial_amount):
	dfNew = df/df.iloc[0]
	dfNew = dfNew * alloc
	dfNew = dfNew * initial_amount
	portfolio_values = dfNew.sum(axis = 1)
	return portfolio_values

def get_daily_returns(portfolio_values):
	daily_returns = portfolio_values.copy()
	daily_returns[1:] = portfolio_values[1:]/(portfolio_values[:-1].values) - 1
	daily_returns = daily_returns[1:]
	return daily_returns

def get_cumm_returns(portfolio_values):
	return (portfolio_values[-1]/portfolio_values[0]) - 1

def get_avg_daily_returns(daily_returns):
	return daily_returns.mean()

def get_std_daily_returns(daily_returns):
	return daily_returns.std()

def get_sharpe_ratio(daily_returns):
	return 15.874*get_avg_daily_returns(daily_returns)/get_std_daily_returns(daily_returns)


def minimize_sharpe_ratio(alloc, df):
	# print "Came here"
	portfolio_values = get_portfolio_values(df, alloc, 1)
	daily_returns = get_daily_returns(portfolio_values)
	sharpe_ratio = get_sharpe_ratio(daily_returns)
	# print alloc, sharpe_ratio
	return -sharpe_ratio

def linear_constraint(x):
	return np.sum(x) - 1
						
# This is the function that will be tested by the autograder 			  		 			     			  	   		   	  			  	
# The student must update this code to properly implement the functionality 			  		 			     			  	   		   	  			  	
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
	syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False): 			  		 			     			  	   		   	  			  	
																				
	# Read in adjusted closing prices for given symbols, date range 			  		 			     			  	   		   	  			  	
	dates = pd.date_range(sd, ed) 			  		 			     			  	   		   	  			  	
	prices_all = get_data(syms, dates)  # automatically adds SPY	  		 			     			  	   		   	  			  	
	prices = prices_all[syms]  # only portfolio symbols 			  		 			     			  	   		   	  			  	
	prices_SPY = prices_all['SPY']  # only SPY, for comparison later

	number_of_stocks = len(syms)
	allocs = [1.0/number_of_stocks]*number_of_stocks
	allocs = np.asarray(allocs)
	bounds = [(0,1)]*number_of_stocks
	constraint = [{'type':'eq', 'fun':linear_constraint}]

	ret = minimize(minimize_sharpe_ratio, allocs, method='SLSQP',
		args=(prices), bounds=bounds, constraints=constraint)

	alloc_new = ret.x
	# alloc_new = [1.0]

	portfolio_values = get_portfolio_values(prices, alloc_new, 1)
	daily_returns = get_daily_returns(portfolio_values)
	average_daily_returns = get_avg_daily_returns(daily_returns)
	cummulative_return = get_cumm_returns(portfolio_values)
	volatility = get_std_daily_returns(daily_returns)
	sharpe_ratio = get_sharpe_ratio(daily_returns)
	# print alloc_new, sharpe_ratio, alloc_new.sum()
																				
	# # Get daily portfolio value
	port_val = portfolio_values # add code here to compute daily portfolio values 			  		 			     			  	   		   	  			  	
																				
	# # # Compare daily portfolio value with SPY using a normalized plot 			  		 			     			  	   		   	  			  	
	if gen_plot: 			  		 			     			  	   		   	  			  	
		# add code to plot here 			  		 			     			  	   		   	  			  	
		df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
		df_temp['SPY'] = df_temp['SPY']/df_temp.iloc[0]['SPY']
 		df_temp.plot()
 		plt.xlabel('Date')
 		plt.ylabel('Price')
 		plt.title('Daily Portfolio Value and SPY')  		 			     			  	   		   	  			  	
		plt.savefig('plot.png')
		plt.close() 		 			     			  	   		   	  			  	
																				
	return alloc_new, cummulative_return, average_daily_returns, volatility, sharpe_ratio #allocs, cr, adr, sddr, sr 			  		 			     			  	   		   	  			  	
																				
def test_code(): 			  		 			     			  	   		   	  			  	
	# This function WILL NOT be called by the auto grader 			  		 			     			  	   		   	  			  	
	# Do not assume that any variables defined here are available to your function/code 			  		 			     			  	   		   	  			  	
	# It is only here to help you set up and test your code 			  		 			     			  	   		   	  			  	
																				
	# Define input parameters 			  		 			     			  	   		   	  			  	
	# Note that ALL of these values will be set to different values by 			  		 			     			  	   		   	  			  	
	# the autograder! 			  		 			     			  	   		   	  			  	
																				
	start_date = dt.datetime(2008,06,01) 			  		 			     			  	   		   	  			  	
	end_date = dt.datetime(2009,06,01) 			  		 			     			  	   		   	  			  	
	symbols = ['IBM', 'X', 'GLD', 'JPM']	
	# symbols = ['HNZ']		  		 			     			  	   		   	  			  	
																				
	# Assess the portfolio 			  		 			     			  	   		   	  			  	
	allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
		syms = symbols, \
		gen_plot = True) 			  		 			     			  	   		   	  			  	
																				
	# Print statistics 			  		 			     			  	   		   	  			  	
	print "Start Date:", start_date 			  		 			     			  	   		   	  			  	
	print "End Date:", end_date 			  		 			     			  	   		   	  			  	
	print "Symbols:", symbols 			  		 			     			  	   		   	  			  	
	print "Allocations:", allocations 			  		 			     			  	   		   	  			  	
	print "Sharpe Ratio:", sr 			  		 			     			  	   		   	  			  	
	print "Volatility (stdev of daily returns):", sddr 			  		 			     			  	   		   	  			  	
	print "Average Daily Return:", adr 			  		 			     			  	   		   	  			  	
	print "Cumulative Return:", cr 			  		 			     			  	   		   	  			  	
																				
if __name__ == "__main__": 			  		 			     			  	   		   	  			  	
	# This code WILL NOT be called by the auto grader 			  		 			     			  	   		   	  			  	
	# Do not assume that it will be called 			  		 			     			  	   		   	  			  	
	test_code() 			  		 			     			  	   		   	  			  	
