import numpy as np

def temp(a):
	a = np.append(a,10)

a = np.zeros(1)
temp(a)
print(a)