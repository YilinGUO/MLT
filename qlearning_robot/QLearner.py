""" 			  		 			     			  	   		   	  			  	
Template for implementing QLearner  (c) 2015 Tucker Balch 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
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
import random as rand 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
class QLearner(object):

    def author(self):
        return 'sgondala3'
 			  		 			     			  	   		   	  			  	
    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False): 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
        self.verbose = verbose 			  		 			     			  	   		   	  			  	
        self.num_actions = num_actions 			  		 			     			  	   		   	  			  	
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.q = np.zeros(shape=(num_states, num_actions))
        self.s = 0
        self.a = 0
        self.replay = []
        # For Dyna
        # T has number of transitions occurred
        self.T = np.full(shape=(num_states, num_actions, num_states),
                         fill_value=0.00001)
        self.R = np.zeros(shape=(num_states, num_actions))

    def getAction(self, s):
        if np.random.random() <= self.rar:
            return np.random.randint(0, self.num_actions)
        return np.argmax(self.q[s])

    def querysetstate(self, s):
        """ 			  		 			     			  	   		   	  			  	
        @summary: Update the state without updating the Q-table 			  		 			     			  	   		   	  			  	
        @param s: The new state 			  		 			     			  	   		   	  			  	
        @returns: The selected action 			  		 			     			  	   		   	  			  	
        """ 			  		 			     			  	   		   	  			  	
        self.s = s
        action = self.getAction(s)
        self.a = action
        if self.verbose: print "s =", s,"a =",action
        return action

    def updateModel(self, s, a, s_prime, r):
        self.T[s, a, s_prime] += 1
        self.R[s, a] = (1 - self.alpha)*self.R[s, a] + self.alpha*r

    def updateQ(self, s, a, s_prime, r):
        self.q[s, a] = (1 - self.alpha) * self.q[s, a] \
                                 + self.alpha * (r + self.gamma * np.max(self.q[s_prime]))

    def queryOnce(self, s_prime, r):
        # Update model
        self.updateModel(self.s, self.a, s_prime, r)

        # Update Q function
        self.updateQ(self.s, self.a, s_prime, r)

        self.replay.append([self.s, self.a, s_prime, r])

        # Set s, a
        self.s = s_prime
        self.a = self.getAction(s_prime)
        self.rar = self.rar*self.radr
        return self.a

    def hallucinate(self):
        state, action, s_prime, reward = self.replay[np.random.choice(len(self.replay))]
        '''
        state = np.random.randint(0, self.num_states)
        action = np.random.randint(0, self.num_actions)
        probs = self.T[state, action].copy()
        probs = probs/np.sum(probs)
        s_prime = np.random.choice(self.num_states, p=probs)
        reward = self.R[state, action]
        '''
        self.updateQ(state, action, s_prime, reward)

    def query(self, s_prime, r):
        """ 			  		 			     			  	   		   	  			  	
        @summary: Update the Q table and return an action 			  		 			     			  	   		   	  			  	
        @param s_prime: The new state 			  		 			     			  	   		   	  			  	
        @param r: The ne state 			  		 			     			  	   		   	  			  	
        @returns: The selected action 			  		 			     			  	   		   	  			  	
        """
        action = self.queryOnce(s_prime, r)
        for _ in range(self.dyna):
            self.hallucinate()
        return action

if __name__=="__main__": 			  		 			     			  	   		   	  			  	
    print "Remember Q from Star Trek? Well, this isn't him" 			  		 			     			  	   		   	  			  	
