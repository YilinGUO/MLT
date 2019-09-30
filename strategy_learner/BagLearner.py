import numpy as np
import RTLearner as rt1
from scipy import stats

class BagLearner(object):
	def __init__(self, learner = None,
				 kwargs = {}, bags = 20, boost = False, verbose = False):
		self.learner = learner
		self.kwargs = kwargs
		self.bags = bags
		self.boost = boost
		self.verbose = verbose
		self.learners = [learner(**kwargs) for i in range(bags)]

	def author(self):
		return 'sgondala3'

	def getData(self, xTrain, yTrain):
		nums = np.random.choice(yTrain.shape[0], yTrain.shape[0], replace=True)
		return xTrain[nums], yTrain[nums]

	def addEvidence(self, xTrain, yTrain):
		for learner in self.learners:
			xData, yData = self.getData(xTrain, yTrain)
			learner.addEvidence(xData, yData)

	def query(self, xTest):
		result = np.zeros(xTest.shape[0])
		for learner in self.learners:
			individualResult = learner.query(xTest)
			result = np.vstack((result, individualResult))
		result = result[1:]
		mode = stats.mode(result, axis=0)[0][0]
		return mode


if __name__=="__main__":
	trainX = np.asarray([[1,2,3,4], [5,6,7,8]])
	trainY = np.asarray([-1, -2])

	bagLearner = BagLearner(learner=rt1.RTLearner, kwargs = {"leaf_size":3}, bags=5)

	bagLearner.addEvidence(trainX, trainY)
	out = bagLearner.query(trainX)
	# print out, trainY