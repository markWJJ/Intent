import tensorflow as tf
import numpy as np

sess=tf.Session()



s=np.array([[[1, 1, 1], [2, 2, 2]],[[3, 3, 3], [4, 4, 4]],[[5, 5, 5], [6, 6, 6]]])
print(s.shape)

s=tf.slice(s, [0, 0, 0], [1, 1, 3])

ss=sess.run(s)

print(ss.shape)