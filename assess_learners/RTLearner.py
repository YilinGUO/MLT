# TODO - Should we add license for this?

import numpy as np

class RTLearner(object):
	def __init__(self, leaf_size = 1, verbose = False):
		self.leaf_size = leaf_size
		self.verbose = verbose
		self.tree = None

	def author(self):
		return 'sgondala3'

	def addEvidenceHelper(self, xTrain, yTrain):
		if yTrain.shape[0] <= self.leaf_size or np.unique(yTrain).shape[0] == 1:
			return np.array([[-1, np.mean(yTrain), None, None]])
		index = np.random.randint(0, xTrain.shape[1])
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
	inf = open("Data/winequality-red.csv")
	data = np.array([map(float, s.strip().split(',')) for s in inf.readlines()])

	# compute how much of the data is training and testing
	train_rows = int(0.6 * data.shape[0])
	test_rows = data.shape[0] - train_rows

	# separate out training and testing data
	trainX = data[:train_rows, 0:-1]
	trainY = data[:train_rows, -1]
	testX = data[train_rows:, 0:-1]
	testY = data[train_rows:, -1]

	dtLearner = RTLearner(leaf_size = 3)

	out = dtLearner.addEvidence(trainX, trainY)
	# print out.shape