from numbapro import *
import numpy
from numbapro.vectorize import GUVectorize

###Equivalent to np.zeros_like
def zeroit(P,PN):
	m, n = P.shape
	for i in range(m):
		for j in range(n):
			PN[i,j] = 0

zeroit = GUVectorize(zeroit, '(m,n)->(m,n)', target='gpu')
zeroit.add(argtypes=[f4[:,:], f4[:,:]])
zeroit.add(argtypes=[f8[:,:], f8[:,:]])
zeroit.add(argtypes=[int32[:,:], int32[:,:]])
zeroit = zeroit.build_ufunc()
