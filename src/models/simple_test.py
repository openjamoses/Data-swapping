import numpy as np

a = np.array([[1,2,3],[4,5,6]])

b = np.array([[3,1,3],[9,0,10]])

#print(a)
#print(b)
#print(np.append(a,b, axis=0))

arr2D = np.array([[11 ,12, 13, 11],
                  [21, 22, 23, 24],
                  [31, 32, 33, 34]])
print(arr2D)

arr2D = np.delete(arr2D, 2, axis=1)
print(arr2D)

a = np.array([0,0,1,3,4,5,5,3,3,3])
b = np.array([0,1,1,3,4,5,5,3,3,3])

print(np.sum(a==0).any())