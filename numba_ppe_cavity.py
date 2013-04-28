from numbapro import *
import numpy as np
from numbapro.vectorize import GUVectorize

###Equivalent to np.zeros_like
def zeroit(P,PN):
	m, n = P.shape
	for i in range(m):
		for j in range(n):
			PN[i,j] = 0

###Implement Von-Neumann boundary conditions
###du/dx = 0 at x = 0, 2, dv/dy=0 at y = 0, p = 0 at y=2
def neumannize(P,PN):
	m, n = P.shape
	PN[:] = P[:]
	for i in range(m):
		PN[i,0]=P[i,1]
		PN[i,n-1]=0 #P[i,n-2]
	for j in range(n):
		PN[0,j]=P[1,j]
		PN[m-1,j]=P[m-2,j]

###Calculate pressure field using pressure-poisson equation
###to ensure that divergence is zero across the cavity
###number of pseudo-time iterations is controlled by global variable 'nit'
def ppe(U,V,P,PN):
	m, n = U.shape
	for count in range(nit):
		for i in range(1,m-1):
			for j in range(1,n-1):
				PN[i, j] = ((P[i+1,j]+P[i-1,j])*dy**2+\
					(P[i,j+1]+P[i,j-1])*dx**2)/(2*(dx**2+dy**2))-\
						dx**2*dy**2/(2*(dx**2+dy**2))*(\
					1/dt*((U[i+1,j]-U[i-1,j])/(2*dx)+\
						(V[i,j+1]-V[i,j-1])/(2*dy))-\
					(((U[i+1,j]-U[i-1,j])/(2*dx))**2)-\
						2*((U[i,j+1]-U[i,j-1])/(2*dy)*\
					(V[i+1,j]-V[i-1,j])/(2*dx))-\
						(((V[i,j+1]-V[i,j-1])/(2*dy))**2))
		G = neumannize(PN)
		PN[:]=G[:]
		P[:]=PN[:]

###Calculate U(n+1) velocity field given U(n), V(n) and P	
def calcU(U,V,P,UN):
	m, n = U.shape
	UN[:]=U[:]
	for i in range(1,m-1):
		for j in range(1,n-1):
			UN[i,j]=U[i,j]-U[i,j]*dt/dx*(U[i,j]-U[i-1,j])-\
				V[i,j]*dt/dy*(U[i,j]-U[i,j-1])-\
			dt/(2*rho*dx)*(P[i+1,j]-P[i-1,j])+\
				nu*(dt/dx**2*(U[i+1,j]-2*U[i,j]+U[i-1,j])+\
			dt/dy**2*(U[i,j+1]-2*U[i,j]+U[i,j-1]))
	###velocity boundary conditions (no slip at all walls except lid)
	for j in range(n):
		UN[0,j]=0
		UN[m-1,j]=0
	for i in range(m):
		UN[i,0]=0
		UN[i,n-1]=1

###Calculate V(n+1) velocity field given U(n), V(n) and P	
def calcV(U,V,P,VN):
	m, n = U.shape
	VN[:]=V[:]
	for i in range(1,m-1):
		for j in range(1,n-1):
			VN[i,j]=V[i,j]-U[i,j]*dt/dx*(V[i,j]-V[i-1,j])-\
				V[i,j]*dt/dy*(V[i,j]-V[i,j-1])-\
			dt/(2*rho*dy)*(P[i,j+1]-P[i,j-1])+\
				nu*(dt/dx**2*(V[i+1,j]-2*V[i,j]+V[i-1,j])+\
			dt/dy**2*(V[i,j+1]-2*V[i,j]+V[i,j-1]))
	###velocity boundary conditions (no slip at all walls)
	for i in range(m):
		VN[i,0]=0
		VN[i,n-1]=0
	for j in range(n):
		VN[0,j]=0
		VN[m-1,j]=0

###Initialize variables, grid size, mesh size, density, viscosity and pseudo-time
nx = 41
ny = 41
dx = 2.0/(nx-1)
dy = 2.0/(ny-1)
dt = .001
nit = 50

rho = 1
nu =.1 

U = np.zeros((ny,nx))
U[:,-1]=1  ###Is this redundant?  Check
V = np.zeros((ny,nx))
P = np.zeros((ny,nx))


###Compile GUfuncs -- use datatype overloading.  Make sure to compile and ufunc referenced elsewhere FIRST
zeroit = GUVectorize(zeroit, '(m,n)->(m,n)')
zeroit.add(argtypes=[f4[:,:], f4[:,:]])
zeroit.add(argtypes=[f8[:,:], f8[:,:]])
zeroit.add(argtypes=[int32[:,:], int32[:,:]])
zeroit = zeroit.build_ufunc()

neumannize = GUVectorize(neumannize, '(a,b)->(a,b)')
neumannize.add(argtypes=[f4[:,:], f4[:,:]])
neumannize.add(argtypes=[f8[:,:], f8[:,:]])
neumannize.add(argtypes=[int32[:,:], int32[:,:]])
neumannize = neumannize.build_ufunc() 

ppe = GUVectorize(ppe, '(m,n),(m,n),(m,n)->(m,n)')
ppe.add(argtypes=[f4[:,:], f4[:,:], f4[:,:], f4[:,:]])
ppe.add(argtypes=[f8[:,:], f8[:,:], f8[:,:], f8[:,:]])
ppe.add(argtypes=[int32[:,:], int32[:,:], int32[:,:], int32[:,:]])
ppe = ppe.build_ufunc() 

calcU = GUVectorize(calcU, '(m,n),(m,n),(m,n)->(m,n)')
calcU.add(argtypes=[f4[:,:], f4[:,:], f4[:,:], f4[:,:]])
calcU.add(argtypes=[f8[:,:], f8[:,:], f8[:,:], f8[:,:]])
calcU.add(argtypes=[int32[:,:], int32[:,:], int32[:,:], int32[:,:]])
calcU = calcU.build_ufunc() 

calcV = GUVectorize(calcV, '(m,n),(m,n),(m,n)->(m,n)')
calcV.add(argtypes=[f4[:,:], f4[:,:], f4[:,:], f4[:,:]])
calcV.add(argtypes=[f8[:,:], f8[:,:], f8[:,:], f8[:,:]])
calcV.add(argtypes=[int32[:,:], int32[:,:], int32[:,:], int32[:,:]])
calcV = calcV.build_ufunc() 

