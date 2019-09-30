import BagLearner as bgl
import LinRegLearner as lrl

class InsaneLearner(object):
	def __init__(self, verbose = False):
		self.learner = bgl.BagLearner(bgl.BagLearner,
						kwargs={"learner":lrl.LinRegLearner, "bags":20}, bags=20)

	def author(self):
		return 'sgondala3'

	def addEvidence(self, xTrain, yTrain):
		self.learner.addEvidence(xTrain, yTrain)

	def query(self, xTest):
		return self.learner.query(xTest)