""" 			  		 			     			  	   		   	  			  	
A simple wrapper for linear regression.  (c) 2015 Tucker Balch 			  		 			     			  	   		   	  			  	
Note, this is NOT a correct DTLearner; Replace with your own implementation. 			  		 			     			  	   		   	  			  	
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
 			  		 			     			  	   		   	  			  	
import numpy as np

class DTLearner(object):
	def __init__(self, leaf_size = 1, verbose = False):
		self.leaf_size = leaf_size
		self.verbose = verbose
		self.tree = None

	def author(self):
		return 'sgondala3'  # replace tb34 with your Georgia Tech username

	def addEvidenceHelper(self, xTrain, yTrain):
		# if xTrain.shape[0] == 9:
		# 	# print np.array2string(xTrain, separator=',')
		# 	print np.array2string(yTrain, separator=',')
		# 	exit(0)
		if yTrain.shape[0] <= self.leaf_size or np.unique(yTrain).shape[0] == 1:
			return np.array([[-1, np.mean(yTrain), None, None]])
		index = np.nanargmax(map(lambda x: abs(np.corrcoef(x, yTrain)[0][1]),xTrain.T))
		value = np.median(xTrain[:, index])
		maskLeft = xTrain[:, index] <= value
		maskRight = xTrain[:, index] > value
		if np.alltrue(maskLeft) or np.alltrue(maskRight):
			return np.array([[-1, np.mean(yTrain), None, None]])
		left = self.addEvidenceHelper(xTrain[maskLeft], yTrain[maskLeft])
		right = self.addEvidenceHelper(xTrain[maskRight], yTrain[maskRight])
		root = np.array([[index, value, 1, left.shape[0] + 1]])
		return np.vstack((root, left, right))

	def addEvidence(self, xTrain, yTrain):
		self.tree = self.addEvidenceHelper(xTrain, yTrain)
		return self.tree

	def queryOne(self, x):
		currentIndex = 0
		node = self.tree[currentIndex]
		while node[0] != -1:
			colIndex = int(node[0])
			if x[colIndex] <= node[1]:
				currentIndex += int(node[2])
				node = self.tree[currentIndex]
			else:
				currentIndex += int(node[3])
				node = self.tree[currentIndex]
		return node[1]

	def query(self, xTest):
		return [self.queryOne(x) for x in xTest]

if __name__ == "__main__":
	inf = open("Data/simple.csv")
	data = np.array([map(float, s.strip().split(',')) for s in inf.readlines()])

	# compute how much of the data is training and testing
	train_rows = int(1 * data.shape[0])
	test_rows = data.shape[0] - train_rows

	# separate out training and testing data
	trainX = data[:train_rows, 0:-1]
	trainY = data[:train_rows, -1]
	testX = data[train_rows:, 0:-1]
	testY = data[train_rows:, -1]

	dtLearner = DTLearner(leaf_size = 1)

	out = dtLearner.addEvidence(trainX, trainY)
	trainOut = dtLearner.query(trainX)
	# print trainOut
	# print len(trainOut), len(testY)
	# sum = 0.0
	# for i in range(len(trainOut)):
	# 	sum += (trainOut[i] == testY[i])
	# print sum/len(trainOut)