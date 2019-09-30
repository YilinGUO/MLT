""" 			  		 			     			  	   		   	  			  	
template for generating data to fool learners (c) 2016 Tucker Balch 			  		 			     			  	   		   	  			  	
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
GT User ID: sgondala3(replace with your User ID)
GT ID: 903388899 (replace with your GT ID)
""" 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
import numpy as np 			  		 			     			  	   		   	  			  	
import math 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
# this function should return a dataset (X and Y) that will work 			  		 			     			  	   		   	  			  	
# better for linear regression than decision trees 			  		 			     			  	   		   	  			  	
def best4LinReg(seed=1489683273): 			  		 			     			  	   		   	  			  	
    np.random.seed(seed) 			  		 			     			  	   		   	  			  	
    X = np.random.random(size= (500,5))
    Y = 1*X[:,0] + 2*X[:,1] + 3*X[:,2] + 4*X[:,3] + 5*X[:,4]
    return X, Y 			  		 			     			  	   		   	  			  	

def best4DT(seed=1489683273):
    np.random.seed(seed)
    X = np.random.random(size=(500, 2))
    Y = np.asarray(map(lambda x : 2 if x[0] > 0.5 else -2, X))
    return X, Y


def author(): 			  		 			     			  	   		   	  			  	
    return 'sgondala3' #Change this to your user ID
 			  		 			     			  	   		   	  			  	
if __name__=="__main__":
    pass
    X1, Y  = best4LinReg(10)
    X2, Y = best4LinReg(10)
    X3, Y = best4LinReg(20)
    X4, Y = best4LinReg(20)
    if np.array_equal(X1, X2):
        print 'Yes'
    else:
        print "No"
    if np.array_equal(X2, X3):
        print "Yes"
    else:
        print "No"
