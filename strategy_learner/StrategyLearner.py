"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch

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

import datetime as dt
import pandas as pd
from indicators import \
    getResolvedAdjustedPrices, getRSI, getPriceSMARatio, getWilliamR, CONSTANTS, benchmark
import numpy as np
import RTLearner as RL
import BagLearner as BGL
from marketsimcode import compute_portvals


class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False, impact=0.0):
        self.verbose = verbose
        self.impact = impact
        self.model = BGL.BagLearner(learner=RL.RTLearner, kwargs = {"leaf_size":5}, bags=20)

    def author(self):
        return 'sgondala3'

    def classify(self, ybuy, ysell, val):
        if val > ybuy and abs(val) >= self.impact*1.5:
            return 1
        elif val < ysell and abs(val) >= self.impact*1.5:
            return -1
        return 0

    def getXVals(self, symbol, sd, ed,
                 lookback_williams, lookback_price_sma, lookback_rsi):
        xVals = getResolvedAdjustedPrices(symbol, pd.date_range(sd, ed))
        xVals.drop(['SPY'], axis=1, inplace=True)

        williamsr = getWilliamR(symbol, sd, ed, lookback_williams)
        xVals = xVals.join(williamsr[CONSTANTS.WILLIAM_R])

        price_sma = getPriceSMARatio(symbol, sd, ed, lookback_price_sma)
        xVals = xVals.join(price_sma[CONSTANTS.PRICE_SMA_RATIO])

        rsi = getRSI(symbol, sd, ed, lookback_rsi)
        xVals = xVals.join(rsi[CONSTANTS.RSI])

        xVals.drop([symbol], axis=1, inplace=True)
        xVals = xVals.values
        return xVals

    # Returns +1/-1/0 for buy/sell/hold
    def getYVals(self, symbol, sd, ed, ndays, ybuy, ysell, length):
        yVals = getResolvedAdjustedPrices(symbol, pd.date_range(sd, ed + dt.timedelta(days = 30)))
        yVals.drop(['SPY'], axis=1, inplace=True)
        yVals = self.get_nday_returns(yVals, ndays)
        yVals = yVals[0:length]
        yVals = np.reshape(yVals, (1,-1))[0]
        classify = np.vectorize(self.classify)
        yVals = classify(ybuy, ysell, yVals)
        return yVals

    # this method should create a RT Learner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000):

        np.random.seed(42)
        # Hyper parameters
        lookback_williams = 3
        lookback_rsi = 3
        lookback_price_sma = 3
        ndays = 1
        ybuy = 0
        ysell = 0

        xVals = self.getXVals(symbol, sd, ed,
                              lookback_williams, lookback_price_sma, lookback_rsi)

        yVals = self.getYVals(symbol, sd, ed, ndays, ybuy, ysell, len(xVals))

        assert np.NaN not in xVals
        assert np.NaN not in yVals
        assert len(xVals) == len(yVals)
        self.model.addEvidence(xVals, yVals)

    def get_nday_returns(self, prices, ndays):
        daily_returns = prices.copy()
        daily_returns = daily_returns/daily_returns.shift(ndays) - 1
        daily_returns = daily_returns[ndays:]
        return daily_returns.values

    def getTrades(self, yVals):
        current_stocks = 0
        retVal = []
        for val in yVals:
            if val == 1:
            # if val > 0 and val > self.impact:
                new_stocks = 1000
                tradeVals = new_stocks - current_stocks
                retVal.append(tradeVals)
                current_stocks = new_stocks
            elif val == -1:
            # elif val < 0 and abs(val) > self.impact:
                new_stocks = -1000
                tradeVals = new_stocks - current_stocks
                retVal.append(tradeVals)
                current_stocks = new_stocks
            else:
                # new_stocks = current_stocks
                tradeVals = 0
                retVal.append(tradeVals)
                # current_stocks = new_stocks
        return retVal


    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):

        np.random.seed(42)
        pricesDf = getResolvedAdjustedPrices(symbol, pd.date_range(sd, ed))
        lookback_williams = lookback_price_sma = lookback_rsi = 3
        xVals = self.getXVals(symbol, sd, ed, lookback_williams, lookback_price_sma, lookback_rsi)
        yPred = self.model.query(xVals)
        trades = self.getTrades(yPred)
        trades_df = pd.DataFrame({CONSTANTS.TRADES:trades}, index=pricesDf.index)
        return trades_df

# def evaluate(symbol, sd, ed):
#     s1 = StrategyLearner()
#
#     lookback_williams = 3
#     lookback_rsi = 3
#     lookback_price_sma = 3
#     ndays = 1
#     ybuy = 0
#     ysell = 0
#
#     s1.addEvidence(symbol, sd, ed)
#     xVals = s1.getXVals(symbol, sd, ed, lookback_williams, lookback_price_sma, lookback_rsi)
#     yPred = s1.model.query(xVals)
#     trades = s1.getTrades(yPred)
#     index = getResolvedAdjustedPrices(symbol, date_range=pd.date_range(sd, ed)).index
#     pd_orders = pd.DataFrame({CONSTANTS.TRADES:trades}, index)
#     predictedValue = compute_portvals(pd_orders, symbol, 100000).ix[ed - dt.timedelta(days=1), CONSTANTS.TOTAL]/100000 - 1
#
#     pd_benchmark = benchmark(symbol, sd, ed)
#     benchmarkValue = compute_portvals(pd_benchmark, symbol, 100000).ix[ed - dt.timedelta(days=1), CONSTANTS.TOTAL]/100000 - 1
#
#     return predictedValue, benchmarkValue

if __name__=="__main__":
    # In Sample
    pass