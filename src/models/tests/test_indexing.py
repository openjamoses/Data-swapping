

import numpy as np

x = np.array([[ 0,  1,  2],
              [ 3,  4,  5],
              [ 6,  7,  8],
              [ 9, 10, 11]])
feature_index = [0,2]
print(np.array(x)[0:, feature_index])

x1 = [1,3,5,3,5,7,8,9,10,0]
x2 = [0,3,4,3,5,7,3,9,0,11]