import pandas as pd
import numpy as np
import datetime as dt
import os
import marketsimcode as marketsim
from util import get_data, plot_data
import matplotlib.pyplot as plt
from indicators import getRSI, getPriceSMARatio, \
	getWilliamR, getResolvedAdjustedPrices, benchmark, CONSTANTS
import matplotlib
matplotlib.use('Agg')


class ManualStrategy(object):
	def __init__(self):
		pass

	def author(self):
		return 'sgondala3'

	def testPolicy(self, symbol = "AAPL",
				   sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000):

		prices = getResolvedAdjustedPrices([symbol], pd.date_range(sd, ed))

		df_RSI = getRSI(symbol, sd, ed)
		df_price_sma = getPriceSMARatio(symbol, sd, ed)
		df_williams = getWilliamR(symbol, sd, ed)

		trades_df = pd.DataFrame(index = prices.index)
		trades_df[CONSTANTS.TRADES] = 0

		date_list = prices.index.values

		current_holdings = 0
		for index, current_date in enumerate(date_list):
			previous_date = date_list[index-1]
			closing_RSI = df_RSI.ix[previous_date, CONSTANTS.RSI]
			closing_price_sma = df_price_sma.ix[previous_date, CONSTANTS.PRICE_SMA_RATIO]
			closing_williams = df_williams.ix[previous_date, CONSTANTS.WILLIAM_R]

			assert not (closing_price_sma == np.NaN or closing_RSI == np.NaN or closing_williams == np.NaN)

			new_holding = 0
			if closing_williams <= -80 and closing_price_sma <= 0.9 and closing_RSI <= 30:
				new_holding = 1000
			if closing_williams >= -20 and closing_price_sma >= 1.1 and closing_RSI >= 70:
				new_holding = -1000

			trade_amount = new_holding - current_holdings
			current_holdings = new_holding
			trades_df.ix[current_date, CONSTANTS.TRADES] = trade_amount

		return trades_df


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


def evaluateInSample():
	ts = ManualStrategy()
	trades_df = ts.testPolicy(CONSTANTS.DEFAULT_SYMBOL,
							  CONSTANTS.IN_SAMPLE_START, CONSTANTS.IN_SAMPLE_END, CONSTANTS.START_CASH)
	port_vals = marketsim.compute_portvals(trades_df, CONSTANTS.DEFAULT_SYMBOL, CONSTANTS.START_CASH)

	port_vals_benchmark = marketsim.compute_portvals(
		benchmark(CONSTANTS.DEFAULT_SYMBOL, CONSTANTS.IN_SAMPLE_START, CONSTANTS.IN_SAMPLE_END),
		CONSTANTS.DEFAULT_SYMBOL,
		CONSTANTS.START_CASH)

	plot_df = pd.DataFrame(index=port_vals.index)
	plot_df["Manual Portfolio"] = port_vals[CONSTANTS.TOTAL]
	plot_df["Benchmark"] = port_vals_benchmark[CONSTANTS.TOTAL]
	plot_df = plot_df / plot_df.ix[plot_df.index.values[0]]
	plot_df.plot(color=['red', 'green'], linewidth=1)

	date_vals = trades_df.index.values
	previous_holdings = 0
	for index, current_date in enumerate(date_vals):
		current_holdings = previous_holdings + trades_df.ix[current_date, CONSTANTS.TRADES]
		if previous_holdings != current_holdings:
			if current_holdings > 0:
				plt.axvline(current_date, color='blue')
			elif current_holdings < 0:
				plt.axvline(current_date, color='black')
		previous_holdings = current_holdings

	plt.savefig('Manual-InSample.png')

	print "In Sample"

	portfolio_values_final = port_vals.sum(axis=1)
	portfolio_values_benchmark_final = port_vals_benchmark.sum(axis=1)
	daily_rets = get_daily_returns(portfolio_values_final)
	daily_rets_benchmark = get_daily_returns(portfolio_values_benchmark_final)


	print "Cumulative returns: ", get_cumm_returns(portfolio_values_benchmark_final), get_cumm_returns(portfolio_values_final)
	print "Stddev: ", get_std_daily_returns(daily_rets_benchmark), get_std_daily_returns(daily_rets)
	print "Mean of daily returns: ", get_avg_daily_returns(daily_rets_benchmark), get_avg_daily_returns(daily_rets)


def evaluateOutSample():
	ts = ManualStrategy()
	trades_df = ts.testPolicy(CONSTANTS.DEFAULT_SYMBOL,
							  CONSTANTS.OUT_SAMPLE_START, CONSTANTS.OUT_SAMPLE_END, CONSTANTS.START_CASH)
	port_vals = marketsim.compute_portvals(trades_df, CONSTANTS.DEFAULT_SYMBOL, CONSTANTS.START_CASH)

	port_vals_benchmark = marketsim.compute_portvals(
		benchmark(CONSTANTS.DEFAULT_SYMBOL, CONSTANTS.OUT_SAMPLE_START, CONSTANTS.OUT_SAMPLE_END),
		CONSTANTS.DEFAULT_SYMBOL,
		CONSTANTS.START_CASH)

	plot_df = pd.DataFrame(index=port_vals.index)
	plot_df["Manual Portfolio"] = port_vals[CONSTANTS.TOTAL]
	plot_df["Benchmark"] = port_vals_benchmark[CONSTANTS.TOTAL]
	plot_df = plot_df / plot_df.ix[plot_df.index.values[0]]
	plot_df.plot(color=['red', 'green'], linewidth=1)

	date_vals = trades_df.index.values
	previous_holdings = 0
	for index, current_date in enumerate(date_vals):
		current_holdings = previous_holdings + trades_df.ix[current_date, CONSTANTS.TRADES]
		if previous_holdings != current_holdings:
			if current_holdings > 0:
				plt.axvline(current_date, color='blue')
			elif current_holdings < 0:
				plt.axvline(current_date, color='black')
		previous_holdings = current_holdings

	plt.savefig('Manual-OutOfSample.png')

	print "Out Sample"
	portfolio_values_final = port_vals.sum(axis=1)
	portfolio_values_benchmark_final = port_vals_benchmark.sum(axis=1)
	daily_rets = get_daily_returns(portfolio_values_final)
	daily_rets_benchmark = get_daily_returns(portfolio_values_benchmark_final)


	print "Cumulative returns: ", get_cumm_returns(portfolio_values_benchmark_final), get_cumm_returns(portfolio_values_final)
	print "Stddev: ", get_std_daily_returns(daily_rets_benchmark), get_std_daily_returns(daily_rets)
	print "Mean of daily returns: ", get_avg_daily_returns(daily_rets_benchmark), get_avg_daily_returns(daily_rets)


if __name__ == "__main__":
	evaluateInSample()
	evaluateOutSample()

