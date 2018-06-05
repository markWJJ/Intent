import numpy as np

a=np.ones(shape=(16,30,50))
b=np.zeros(shape=(16,30,1))

s=np.multiply(a,b)

print(s.shape)