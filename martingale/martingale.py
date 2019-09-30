"""Assess a betting strategy.                                                                               
                                                                                
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

from __future__ import print_function                                                                                
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def author():                                                                               
        return 'sgondala3' # replace tb34 with your Georgia Tech username.                                                                              
                                                                                
def gtid():                                                                                 
    return 903388899 # replace with your GT ID number                                                                               
                                                                                
def get_spin_result(win_prob):                                                                              
    result = False                                                                              
    if np.random.random() <= win_prob:                                                                              
        result = True                                                                               
    return result

def populate_array(array):
    while array.size <= 1000:
        array = np.append(array, array[-1])
    return array

def get_episode_winnings(win_prob):
    returnVal = np.zeros(1)
    episode_winnings = 0
    spin_number = 1
    while episode_winnings < 80 and spin_number <= 1000:
        won = False
        bet_amount = 1
        while not won and spin_number <= 1000:
            won = get_spin_result(win_prob)
            if won == True:
                episode_winnings += bet_amount
            else:
                episode_winnings -= bet_amount
                bet_amount *= 2
            returnVal = np.append(returnVal, episode_winnings)
            spin_number += 1
    returnVal = populate_array(returnVal)
    return returnVal

def get_episode_winnings_with_limit(win_prob, amount):
    returnVal = np.zeros(1)
    episode_winnings = 0
    amount_left = amount
    spin_number = 1
    while episode_winnings < 80 and amount_left >= 0 and spin_number <= 1000:
        won = False
        bet_amount = 1
        while not won and amount_left >= 0 and spin_number <= 1000:
            won = get_spin_result(win_prob)
            if won == True:
                episode_winnings += bet_amount
                amount_left += bet_amount
            else:
                episode_winnings -= bet_amount
                amount_left -= bet_amount
                bet_amount = min(2*bet_amount, amount_left)
            returnVal = np.append(returnVal, episode_winnings)
            spin_number += 1
    returnVal = populate_array(returnVal)
    return returnVal

def plot(df, title):
    plt.close()
    df.plot()
    plt.xlim(0,300)
    plt.ylim(-256,100)
    plt.title(title)
    plt.xlabel("Spin Number")
    plt.ylabel("Estimated earnings in $")
    plt.savefig(title)


def figure1(win_prob, limit, title):
    df = pd.DataFrame()
    for i in range(10):
        episode_winnings = None
        if limit == 0:
            episode_winnings = get_episode_winnings(win_prob)
        else:
            episode_winnings = get_episode_winnings_with_limit(win_prob, limit)
        name = "Simulation " + str(i)
        df[name] = episode_winnings
    plot(df, title)


def figure2_and_3(win_prob, limit, title1, title2):
    df = pd.DataFrame()
    for i in range(1000):
        episode_winnings = None
        if limit == 0:
            episode_winnings = get_episode_winnings(win_prob)
        else:
            episode_winnings = get_episode_winnings_with_limit(win_prob, limit)
        df[i] = episode_winnings

    df = df.T
    a = df.mean()
    b = df.std()
    dfNew = pd.DataFrame()
    dfNew['Mean'] = a
    dfNew['Mean - std'] = a - b
    dfNew['Mean + std'] = a + b
    plot(dfNew, title1)
    
    df3 = pd.DataFrame()
    a = df.median()
    df3['Median'] = a
    df3['Median - std'] = a - b
    df3['Median + std'] = a + b
    plot(df3, title2)

                                                                             
def test_code():                                                                                
    win_prob = 0.47368 # set appropriately to the probability of a win                                                                              
    np.random.seed(gtid()) # do this only once                                                                              
    
    figure1(win_prob, 0, "Figure 1")
    figure2_and_3(win_prob, 0, "Figure 2", "Figure 3")
    figure2_and_3(win_prob, 256, "Figure 4", "Figure 5")

if __name__ == "__main__":                                                                              
    test_code()                                                                                 
