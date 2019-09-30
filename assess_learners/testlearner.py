""" 			  		 			     			  	   		   	  			  	
Test a learner.  (c) 2015 Tucker Balch 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
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
""" 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
import numpy as np 			  		 			     			  	   		   	  			  	
import math 			  		 			     			  	   		   	  			  	
import LinRegLearner as lrl
import DTLearner
import RTLearner
import InsaneLearner
import BagLearner
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

if __name__=="__main__": 			  		 			     			  	   		   	  			  	
    if len(sys.argv) != 2: 			  		 			     			  	   		   	  			  	
        print "Usage: python testlearner.py <filename>" 			  		 			     			  	   		   	  			  	
        sys.exit(1) 			  		 			     			  	   		   	  			  	
    inf = open(sys.argv[1])

    data = np.genfromtxt(inf, delimiter=',')
    # f = util.get_learner_data_file(sys.argv[1])
    # data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])

    # data = np.array([s.strip().split(',') for s in inf.readlines()])

    # data = np.genfromtxt(f,delimiter=',')
    if 'Istanbul.csv' in sys.argv[1]:
        print "Came here"
        data = data[1:, 1:]

    data.astype(float)

    # compute how much of the data is training and testing 			  		 			     			  	   		   	  			  	
    train_rows = int(0.6* data.shape[0]) 			  		 			     			  	   		   	  			  	
    test_rows = data.shape[0] - train_rows 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
    # separate out training and testing data 			  		 			     			  	   		   	  			  	
    trainX = data[:train_rows,0:-1] 			  		 			     			  	   		   	  			  	
    trainY = data[:train_rows,-1] 			  		 			     			  	   		   	  			  	
    testX = data[train_rows:,0:-1] 			  		 			     			  	   		   	  			  	
    testY = data[train_rows:,-1] 			  		 			     			  	   		   	  			  	

    # Part 1
    dtTrainRMSE = []
    dtTestRMSE = []

    for leaf_size in range(1,10):
        learner = DTLearner.DTLearner(leaf_size = leaf_size, verbose = False) # create a LinRegLearner
        learner.addEvidence(trainX, trainY) # train it

        # evaluate in sample
        predY = learner.query(trainX) # get the predictions
        rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])

        dtTrainRMSE.append(rmse)

        # evaluate out of sample
        predY = learner.query(testX) # get the predictions
        rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        dtTestRMSE.append(rmse)

    plt.plot(range(1,10), dtTrainRMSE,  label='Train')
    plt.plot(range(1,10), dtTestRMSE, label='Test')
    plt.xlabel('Number of Leafs')
    plt.ylabel('RMSE')
    plt.title('Error graph for DT')
    plt.legend(loc='best')
    plt.savefig('DT.png')
    plt.close()

    # Part 2
    bagTrainRMSE = []
    bagTestRMSE = []

    X=range(1,20)
    for leaf_size in X:
        learner = BagLearner.BagLearner(learner = DTLearner.DTLearner,
                                        kwargs={'leaf_size':leaf_size}, bags=20) # create a LinRegLearner
        learner.addEvidence(trainX, trainY) # train it

        # evaluate in sample
        predY = learner.query(trainX) # get the predictions
        rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])

        bagTrainRMSE.append(rmse)

        # evaluate out of sample
        predY = learner.query(testX) # get the predictions
        rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        bagTestRMSE.append(rmse)

    # print bagTrainRMSE, dtTestRMSE
    plt.plot(X, bagTrainRMSE,  label='Train')
    plt.plot(X, bagTestRMSE, label='Test')
    plt.xlabel('Number of Leafs')
    plt.ylabel('RMSE')
    plt.title('Error graph for Bagging')
    plt.legend(loc='best')
    plt.savefig('Bag.png')
    plt.close()

    # Part 3
    dtTrainRMSE = []
    dtTestRMSE = []
    rtTrainRMSE = []
    rtTestRMSE = []

    dtConstructionTime = []
    rtConstructionTime = []
    dtLength = []
    rtLength = []

    dtQueryTime = []
    rtQueryTime = []

    X=range(1,10)
    for leaf_size in X:
        dt = DTLearner.DTLearner(leaf_size= leaf_size) # create a LinRegLearner
        start = time.time()
        dt.addEvidence(trainX, trainY) # train it
        end = time.time()

        dtConstructionTime.append(end - start)
        dtLength.append(dt.tree.shape[0])

        predY = dt.query(trainX)
        rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
        dtTrainRMSE.append(rmse)

        start = time.time()
        predY = dt.query(testX)
        end = time.time()
        dtQueryTime.append(end - start)
        rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        dtTestRMSE.append(rmse)

        rt = RTLearner.RTLearner(leaf_size=leaf_size)
        start = time.time()
        rt.addEvidence(trainX,trainY)
        end = time.time()

        rtConstructionTime.append(end - start)
        rtLength.append(rt.tree.shape[0])

        predY = rt.query(trainX)
        rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
        rtTrainRMSE.append(rmse)

        start = time.time()
        predY = rt.query(testX)
        end = time.time()
        rtQueryTime.append(end - start)
        rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        rtTestRMSE.append(rmse)

    # print dtTrainRMSE, dtTestRMSE, rtTrainRMSE, rtTestRMSE
    print dtConstructionTime, rtConstructionTime
    print dtLength, rtLength
    print dtQueryTime, rtQueryTime
    plt.plot(X, dtTrainRMSE,  label='Decision Tree Train')
    plt.plot(X, dtTestRMSE, label='Decision Tree Test')
    plt.plot(X, rtTrainRMSE, label='Random Tree Train')
    plt.plot(X, rtTestRMSE, label='Random Tree Test')
    plt.xlabel('Number of Leafs')
    plt.ylabel('RMSE')
    plt.title('Error graph for Exp3')
    plt.legend(loc='best')
    plt.savefig('Comparison.png')
    plt.close()

    plt.plot(X, dtConstructionTime, label='Decision Tree Construction Time')
    plt.plot(X, rtConstructionTime, label='Random Tree Construction Time')
    plt.xlabel('Leaf Size')
    plt.ylabel('Time in s')
    plt.title('Construction times')
    plt.legend(loc='best')
    plt.savefig('ConstructionTimes.png')
    plt.close()

    plt.plot(X, dtQueryTime, label='Decision Tree Query Time')
    plt.plot(X, rtQueryTime, label='Random Tree Query Time')
    plt.xlabel('Leaf Size')
    plt.ylabel('Time in s')
    plt.title('Query times')
    plt.legend(loc='best')
    plt.savefig('QueryTimes.png')
    plt.close()
