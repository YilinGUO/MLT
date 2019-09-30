import pandas as pd
import numpy as np
import datetime as dt
import os
import marketsimcode as marketsim
from util import get_data, plot_data
import matplotlib.pyplot as plt
from indicators import CONSTANTS, benchmark
import matplotlib
matplotlib.use('Agg')


class TheoreticallyOptimalStrategy(object):
	def __init__(self):
		pass

	def author(self):
		return 'sgondala3'

	def testPolicy(self, symbol = "AAPL",
				   sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31),
				   sv = 100000):
		prices = get_data([symbol], pd.date_range(sd, ed))
		trades = pd.DataFrame(index=prices.index)
		trades[CONSTANTS.TRADES] = 0

		current_position = 0
		date_list = prices.index.values
		for index in range(len(date_list[:-1])):
			date_today = date_list[index]
			date_tomorrow = date_list[index+1]
			price_today = prices.ix[date_today, symbol]
			price_tomorrow = prices.ix[date_tomorrow, symbol]
			if price_tomorrow > price_today:
				new_position = 1000
				trades.ix[date_today, CONSTANTS.TRADES] = new_position - current_position
				current_position = new_position
			else:
				new_position = -1000
				trades.ix[date_today, CONSTANTS.TRADES] = new_position - current_position
				current_position = new_position
		return trades

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


if __name__ == "__main__":
	ts = TheoreticallyOptimalStrategy()
	trades_df = ts.testPolicy(CONSTANTS.DEFAULT_SYMBOL,
							  CONSTANTS.IN_SAMPLE_START, CONSTANTS.IN_SAMPLE_END, CONSTANTS.START_CASH)
	port_vals = marketsim.compute_portvals(trades_df, CONSTANTS.DEFAULT_SYMBOL, CONSTANTS.START_CASH, 0, 0)
	port_vals_benchmark = marketsim.compute_portvals(
		benchmark(CONSTANTS.DEFAULT_SYMBOL, CONSTANTS.IN_SAMPLE_START, CONSTANTS.IN_SAMPLE_END),
		CONSTANTS.DEFAULT_SYMBOL,
		CONSTANTS.START_CASH)

	plot_df = pd.DataFrame(index = port_vals.index)
	plot_df["Optimal Portfolio"] = port_vals[CONSTANTS.TOTAL]
	plot_df["Benchmark"] = port_vals_benchmark[CONSTANTS.TOTAL]
	plot_df = plot_df/plot_df.ix[plot_df.index.values[0]]
	plot_df.plot(colors = ['red', 'green'])
	plt.savefig('TOP.png')

	portfolio_values_final = port_vals.sum(axis=1)
	portfolio_values_benchmark_final = port_vals_benchmark.sum(axis=1)
	daily_rets = get_daily_returns(portfolio_values_final)
	daily_rets_benchmark = get_daily_returns(portfolio_values_benchmark_final)


	print "Cumulative returns: ", get_cumm_returns(portfolio_values_benchmark_final), get_cumm_returns(portfolio_values_final)
	print "Stddev: ", get_std_daily_returns(daily_rets_benchmark), get_std_daily_returns(daily_rets)
	print "Mean of daily returns: ", get_avg_daily_returns(daily_rets_benchmark), get_avg_daily_returns(daily_rets)


