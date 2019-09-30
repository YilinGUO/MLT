import datetime as dt
import pandas as pd
from indicators import \
    getResolvedAdjustedPrices, getRSI, getPriceSMARatio, getWilliamR, CONSTANTS, benchmark
import numpy as np
import RTLearner as RL
import BagLearner as BGL
from marketsimcode import compute_portvals
import ManualStrategy
import StrategyLearner
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def author():
	return 'sgondala3'


def getNumTrades(trades_df):
	vals = trades_df.values
	daysTraded = np.sum(np.asarray(map(lambda x: 1 if x != 0 else 0, vals)))
	return daysTraded

def evaluate(symbol, sd, ed, startVal):
	port_vals_benchmark = compute_portvals(
		benchmark(symbol, sd, ed),symbol, startVal)

	indices = port_vals_benchmark.index.values

	impact = 0
	ts = StrategyLearner.StrategyLearner(impact=0)
	ts.addEvidence(symbol, sd, ed, startVal)
	trades_df = ts.testPolicy(symbol, sd, ed, startVal)
	print "Number of trades with impact: ", impact, " is ", getNumTrades(trades_df)
	port_vals_strategy0 = compute_portvals(trades_df, symbol, startVal, impact=0)
	print "Cumulative returns with impact: ", impact, "are ", \
		port_vals_strategy0.ix[indices[-1], CONSTANTS.TOTAL]/port_vals_strategy0.ix[indices[0], CONSTANTS.TOTAL]

	impact = 0.010
	ts = StrategyLearner.StrategyLearner(impact=impact)
	ts.addEvidence(symbol, sd, ed, startVal)
	trades_df = ts.testPolicy(symbol, sd, ed, startVal)
	print "Number of trades with impact: ", impact, " is ", getNumTrades(trades_df)
	port_vals_strategy5 = compute_portvals(trades_df, symbol, startVal, impact=impact)
	print "Cumulative returns with impact: ", impact, "are ", \
		port_vals_strategy5.ix[indices[-1], CONSTANTS.TOTAL] / port_vals_strategy5.ix[indices[0], CONSTANTS.TOTAL]

	impact = 0.015
	ts = StrategyLearner.StrategyLearner(impact=impact)
	ts.addEvidence(symbol, sd, ed, startVal)
	trades_df = ts.testPolicy(symbol, sd, ed, startVal)
	print "Number of trades with impact: ", impact, " is ", getNumTrades(trades_df)
	port_vals_strategy15 = compute_portvals(trades_df, symbol, startVal, impact=impact)
	print "Cumulative returns with impact: ", impact, "are ", \
		port_vals_strategy15.ix[indices[-1], CONSTANTS.TOTAL] / port_vals_strategy15.ix[indices[0], CONSTANTS.TOTAL]

	plot_df = pd.DataFrame(index=port_vals_benchmark.index)
	plot_df["Benchmark"] = port_vals_benchmark[CONSTANTS.TOTAL]
	plot_df["Impact 0"] = port_vals_strategy0[CONSTANTS.TOTAL]
	plot_df["Impact 10"] = port_vals_strategy5[CONSTANTS.TOTAL]
	plot_df["Impact 15"] = port_vals_strategy15[CONSTANTS.TOTAL]
	plot_df = plot_df / plot_df.ix[plot_df.index.values[0]]
	plot_df.plot(color=['red', 'green', 'blue', 'black'], linewidth=1)

	plt.savefig('experiment2.png')



if __name__ == "__main__":
	# In Sample
	symbol = "JPM"
	sd = dt.datetime(2008, 1, 1)
	ed = dt.datetime(2009, 12, 31)
	startVal = 100000
	evaluate(symbol, sd, ed, startVal)