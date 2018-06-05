import random
import numpy as np
s=list(range(0,30))
random.shuffle(s)


ss=np.ones(shape=[3,7])

ss_=np.sum(ss,1)
print(ss_)

