import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use('Agg')

def author():
	return 'sgondala3'

# Constants
class CONSTANTS:
	LOOKBACK_RSI = 3
	LOOKBACK_PRICE_SMA = 3
	LOOKBACK_WILLIAMS = 3
	EPSILON = 0.0001
	RSI = "RSI"
	AVERAGE_GAIN = "AVERAGE_GAIN"
	AVERAGE_LOSS = "AVERAGE_LOSS"
	RS = "RS"
	PRICE = "PRICE"
	SMA = "SMA"
	SPY = "SPY"
	PRICE_SMA_RATIO = "PRICE_SMA_RATIO"
	HIGHEST_HIGH = "HIGHEST_HIGH"
	LOWEST_LOW = "LOWEST_LOW"
	WILLIAM_R = "WILLIAM_R"
	TRADES = "TRADES"
	CASH = "CASH"
	TOTAL = "TOTAL"
	IN_SAMPLE_START = dt.datetime(2008, 1, 1)
	IN_SAMPLE_END = dt.datetime(2009, 12, 31)
	OUT_SAMPLE_START = dt.datetime(2010, 1, 1)
	OUT_SAMPLE_END = dt.datetime(2011, 12, 31)
	DEFAULT_SYMBOL = "JPM"
	START_CASH = 100000

# Common functions
def getResolvedAdjustedPrices(symbols, date_range):
	if not isinstance(symbols, list):
		symbols = [symbols]
	prices = get_data(symbols, date_range)
	prices.bfill(inplace=True)
	prices.ffill(inplace=True)
	return prices


def benchmark(symbol=CONSTANTS.DEFAULT_SYMBOL,
			   sd=CONSTANTS.IN_SAMPLE_START, ed=CONSTANTS.IN_SAMPLE_END,
			   sv=CONSTANTS.START_CASH):

	prices = getResolvedAdjustedPrices([symbol], pd.date_range(sd, ed))
	trades_df = pd.DataFrame(index=prices.index)
	first_date_of_trade = prices.index.values[0]
	trades_df[CONSTANTS.TRADES] = 0

	trades_df.ix[first_date_of_trade, CONSTANTS.TRADES] = 1000
	return trades_df

# RSI[day] calculates the RSI at the end of the day,
# and includes that day's closing values
def getRSI(symbol=CONSTANTS.DEFAULT_SYMBOL,
		   sd=CONSTANTS.IN_SAMPLE_START, ed=CONSTANTS.IN_SAMPLE_END,
		   lookback_rsi = CONSTANTS.LOOKBACK_RSI):
	# Dummy start date to calculate stats on day 1.
	dummy_start_date = sd - dt.timedelta(lookback_rsi * 2 + 30)
	prices = getResolvedAdjustedPrices([symbol], pd.date_range(dummy_start_date, ed))
	rsi = pd.DataFrame(index=prices.index)
	rsi[CONSTANTS.RSI] = np.NaN
	rsi[CONSTANTS.AVERAGE_GAIN] = np.NaN
	rsi[CONSTANTS.AVERAGE_LOSS] = np.NaN
	rsi[CONSTANTS.PRICE] = prices[symbol]

	date_list = rsi.index.values
	for index, current_date in enumerate(date_list):
		if index < lookback_rsi:
			continue

		average_gain = 0.0
		average_loss = 0.0

		for i in range(lookback_rsi):
			current_day_value = prices.ix[date_list[index - lookback_rsi + i + 1], symbol]
			prev_day_value = prices.ix[date_list[index - lookback_rsi + i], symbol]
			difference = current_day_value - prev_day_value

			if difference > 0.0:
				average_gain += difference
			else:
				average_loss += abs(difference)

		assert average_loss >= 0.0
		assert average_gain >= 0.0
		average_loss /= lookback_rsi
		average_gain /= lookback_rsi

		rsi.ix[current_date, CONSTANTS.AVERAGE_GAIN] = average_gain
		rsi.ix[current_date, CONSTANTS.AVERAGE_LOSS] = average_loss

		if abs(average_loss-0) < CONSTANTS.EPSILON:
			rsi.ix[current_date, CONSTANTS.RSI] = 100.0
			rsi.ix[current_date, CONSTANTS.RS] = np.NaN
		else:
			rsi.ix[current_date, CONSTANTS.RS] = average_gain/average_loss
			rsi.ix[current_date, CONSTANTS.RSI] = \
				100.0 - 100.0 / (1 + rsi.ix[current_date, CONSTANTS.RS])

	return rsi.ix[sd:ed]

# At the end of the day, includes value of that day
def getPriceSMARatio(symbol=CONSTANTS.DEFAULT_SYMBOL,
					 sd=CONSTANTS.IN_SAMPLE_START, ed=CONSTANTS.IN_SAMPLE_END,
					 lookback_price_sma = CONSTANTS.LOOKBACK_PRICE_SMA):
	dummy_start_date = sd - dt.timedelta(lookback_price_sma * 2 + 30)
	prices = getResolvedAdjustedPrices([symbol], pd.date_range(dummy_start_date, ed))
	price_sma_df = pd.DataFrame(index=prices.index)

	price_sma_df[CONSTANTS.PRICE] = prices[symbol]
	price_sma_df[CONSTANTS.SMA] = np.NaN
	price_sma_df[CONSTANTS.PRICE_SMA_RATIO] = np.NaN

	date_list = prices.index.values
	for index, current_date in enumerate(date_list):
		if index < lookback_price_sma:
			continue

		assert date_list[index] == current_date
		price_sma_df.ix[current_date, CONSTANTS.SMA] = \
			price_sma_df.ix[date_list[index - lookback_price_sma + 1]:date_list[index],
			CONSTANTS.PRICE].mean()
		price_sma_df.ix[current_date, CONSTANTS.PRICE_SMA_RATIO] = \
			prices.ix[current_date, symbol]/price_sma_df.ix[current_date, CONSTANTS.SMA]

	return price_sma_df.ix[sd:ed]

# At the end of the day, includes value of that day
def getWilliamR(symbol=CONSTANTS.DEFAULT_SYMBOL,
				sd=CONSTANTS.IN_SAMPLE_START, ed=CONSTANTS.IN_SAMPLE_END,
				lookback_williams = CONSTANTS.LOOKBACK_WILLIAMS):
	dummy_start_date = sd - dt.timedelta(lookback_williams * 2 + 30)
	prices = getResolvedAdjustedPrices([symbol], pd.date_range(dummy_start_date, ed))

	william_r_df = pd.DataFrame(index=prices.index)

	william_r_df[CONSTANTS.HIGHEST_HIGH] = np.NaN
	william_r_df[CONSTANTS.LOWEST_LOW] = np.NaN
	william_r_df[CONSTANTS.WILLIAM_R] = np.NaN
	william_r_df[CONSTANTS.PRICE] = prices[symbol]

	date_list = prices.index.values
	for index, current_date in enumerate(date_list):
		if index < lookback_williams + 2:
			continue

		assert date_list[index] == current_date
		william_r_df.ix[current_date, CONSTANTS.HIGHEST_HIGH] = \
			prices.ix[date_list[index - lookback_williams + 1]: current_date, symbol].max()
		william_r_df.ix[current_date, CONSTANTS.LOWEST_LOW] = \
			prices.ix[date_list[index - lookback_williams + 1]: current_date, symbol].min()
		william_r_df.ix[current_date, CONSTANTS.WILLIAM_R] = \
			(william_r_df.ix[current_date, CONSTANTS.HIGHEST_HIGH] - prices.ix[current_date, symbol])/\
			(william_r_df.ix[current_date, CONSTANTS.HIGHEST_HIGH] - william_r_df.ix[current_date, CONSTANTS.LOWEST_LOW]) * -100.0

	return william_r_df.ix[sd:ed]


def plot_price_sma(symbol=CONSTANTS.DEFAULT_SYMBOL,
				   sd=CONSTANTS.IN_SAMPLE_START, ed=CONSTANTS.IN_SAMPLE_END):
	df_price_sma = getPriceSMARatio(symbol, sd, ed)

	G = gridspec.GridSpec(3, 1)

	axes1 = plt.subplot(G[:-1, :])
	plt.plot(df_price_sma.index.values,
			 df_price_sma[CONSTANTS.PRICE], label=CONSTANTS.PRICE, linewidth=1)
	plt.plot(df_price_sma.index.values,
			 df_price_sma[CONSTANTS.SMA], label=CONSTANTS.SMA, linewidth=1)
	plt.legend()

	axes2 = plt.subplot(G[-1, :])
	plt.plot(df_price_sma.index.values,
			 df_price_sma[CONSTANTS.PRICE_SMA_RATIO],
			 label=CONSTANTS.PRICE_SMA_RATIO, linewidth=1)
	plt.hlines(1.1, CONSTANTS.IN_SAMPLE_START, CONSTANTS.IN_SAMPLE_END)
	plt.hlines(0.9, CONSTANTS.IN_SAMPLE_START, CONSTANTS.IN_SAMPLE_END)
	plt.legend()

	plt.tight_layout()
	plt.savefig('Price-SMA-Indicator.png')
	plt.close()

def plot_williams(symbol=CONSTANTS.DEFAULT_SYMBOL,
				   sd=CONSTANTS.IN_SAMPLE_START, ed=CONSTANTS.IN_SAMPLE_END):
	df_williams = getWilliamR(symbol, sd, ed)

	G = gridspec.GridSpec(3, 1)

	axes1 = plt.subplot(G[:-1, :])
	plt.plot(df_williams.index.values,
			 df_williams[CONSTANTS.HIGHEST_HIGH],
			 label=CONSTANTS.HIGHEST_HIGH, color='red', linewidth=1)
	plt.plot(df_williams.index.values,
			 df_williams[CONSTANTS.LOWEST_LOW],
			 label=CONSTANTS.LOWEST_LOW, color='blue', linewidth=1)
	plt.plot(df_williams.index.values,
			 df_williams[CONSTANTS.PRICE],
			 label=CONSTANTS.PRICE, color='green', linewidth=1)
	plt.legend()

	axes2 = plt.subplot(G[-1, :])
	plt.plot(df_williams.index.values,
			 df_williams[CONSTANTS.WILLIAM_R], label=CONSTANTS.WILLIAM_R, linewidth=1)
	plt.hlines(-80, CONSTANTS.IN_SAMPLE_START, CONSTANTS.IN_SAMPLE_END)
	plt.hlines(-20, CONSTANTS.IN_SAMPLE_START, CONSTANTS.IN_SAMPLE_END)
	plt.legend()

	plt.tight_layout()
	plt.savefig('WilliamsR-Indicator.png')
	plt.close()

def plot_rsi(symbol=CONSTANTS.DEFAULT_SYMBOL,
				   sd=CONSTANTS.IN_SAMPLE_START, ed=CONSTANTS.IN_SAMPLE_END):
	df_rsi = getRSI(symbol, sd, ed)

	G = gridspec.GridSpec(3, 1)

	axes1 = plt.subplot(G[0, :])
	plt.plot(df_rsi.index.values,
			 df_rsi[CONSTANTS.PRICE], label=CONSTANTS.PRICE,linewidth=1)
	plt.legend()

	axes1 = plt.subplot(G[1, :])
	plt.plot(df_rsi.index.values,
			 df_rsi[CONSTANTS.AVERAGE_GAIN], label=CONSTANTS.AVERAGE_GAIN,linewidth=1)
	plt.plot(df_rsi.index.values,
			 df_rsi[CONSTANTS.AVERAGE_LOSS], label=CONSTANTS.AVERAGE_LOSS,linewidth=1)
	plt.legend()

	axes2 = plt.subplot(G[-1, :])
	plt.plot(df_rsi.index.values, df_rsi[CONSTANTS.RSI],
			 label=CONSTANTS.RSI, linewidth=1)
	plt.hlines(70, CONSTANTS.IN_SAMPLE_START, CONSTANTS.IN_SAMPLE_END)
	plt.hlines(30, CONSTANTS.IN_SAMPLE_START, CONSTANTS.IN_SAMPLE_END)
	plt.legend()

	plt.tight_layout()
	plt.savefig('RSI-Indicator.png')
	plt.close()


if __name__ == "__main__":
	plot_price_sma(CONSTANTS.DEFAULT_SYMBOL, CONSTANTS.IN_SAMPLE_START,
				   CONSTANTS.IN_SAMPLE_END, CONSTANTS.LOOKBACK_PRICE_SMA)
	plot_williams(CONSTANTS.DEFAULT_SYMBOL, CONSTANTS.IN_SAMPLE_START,
				  CONSTANTS.IN_SAMPLE_END, CONSTANTS.LOOKBACK_WILLIAMS)
	plot_rsi(CONSTANTS.DEFAULT_SYMBOL, CONSTANTS.IN_SAMPLE_START,
			 CONSTANTS.IN_SAMPLE_END, CONSTANTS.LOOKBACK_RSI)
