import numpy as np
from math import sqrt, pi

dstress_dA = np.ones((10,10,1))
#dstress_dA[0][0][1] = 6
# print dstress_dA
# #print dstress_dA
# print np.shape(dstress_dA)
# print np.shape(dstress_dA[:][:])
# temp = np.zeros((10,10))
# for i in range(0,10):
#     temp[i] = dstress_dA[i]
dstress_dA = np.reshape(dstress_dA,(10,10), order = 'C')
print np.shape(dstress_dA)




