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

def evaluate(symbol, sd, ed, startVal):
	ts = ManualStrategy.ManualStrategy()
	trades_df = ts.testPolicy(symbol,sd, ed, startVal)
	port_vals = compute_portvals(trades_df, symbol, startVal)

	indices = port_vals.index.values

	print "Cumulative returns manual strategy is: ", \
		port_vals.ix[indices[-1], CONSTANTS.TOTAL]/port_vals.ix[indices[0], CONSTANTS.TOTAL]

	port_vals_benchmark = compute_portvals(
		benchmark(symbol, sd, ed),symbol, startVal)

	print "Cumulative returns benchmark is: ", \
		port_vals_benchmark.ix[indices[-1], CONSTANTS.TOTAL]/port_vals_benchmark.ix[indices[0], CONSTANTS.TOTAL]

	ts = StrategyLearner.StrategyLearner(impact=0)
	ts.addEvidence(symbol, sd, ed, startVal)
	trades_df = ts.testPolicy(symbol, sd, ed, startVal)
	port_vals_strategy = compute_portvals(trades_df, symbol, startVal)
	print "Cumulative returns strategy learner is: ", \
		port_vals_strategy.ix[indices[-1], CONSTANTS.TOTAL]/port_vals_strategy.ix[indices[0], CONSTANTS.TOTAL]


	plot_df = pd.DataFrame(index=port_vals.index)
	plot_df["Manual Portfolio"] = port_vals[CONSTANTS.TOTAL]
	plot_df["Benchmark"] = port_vals_benchmark[CONSTANTS.TOTAL]
	plot_df["Strategy Learner"] = port_vals_strategy[CONSTANTS.TOTAL]
	plot_df = plot_df / plot_df.ix[plot_df.index.values[0]]
	plot_df.plot(color=['red', 'green', 'blue'], linewidth=1)

	plt.savefig('experiment1.png')

if __name__ == "__main__":
	# In Sample
	symbol = "JPM"
	sd = dt.datetime(2008, 1, 1)
	ed = dt.datetime(2009, 12, 31)
	startVal = 100000
	evaluate(symbol, sd, ed, startVal)