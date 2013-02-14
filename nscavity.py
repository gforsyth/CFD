##Cavity Flow Navier Stokes
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
#import matplotlib.pyplot as plt
import numpy as np
#import os

#plt.ion()

##variable declarations
nx = 31
ny = 31
nt = 100
c = 1
dx = 2.0/(nx-1)
dy = 2.0/(ny-1)
dt = dx*dy

##physical variables
rho = 1
nu = 1

##initial conditions
u = np.zeros((ny,nx)) ##create a XxY vector of 0's
un = np.zeros((ny,nx)) ##create a XxY vector of 0's

v = np.zeros((ny,nx)) ##create a XxY vector of 0's
vn = np.zeros((ny,nx)) ##create a XxY vector of 0's

p = np.zeros((ny,nx)) ##create a XxY vector of 0's
pn = np.zeros((ny,nx)) ##create a XxY vector of 0's


for n in range(nt):
	un[:] = u[:]
	vn[:] = v[:]
	pn[:] = p[:]

	u[1:-1,1:-1] = un[1:-1,1:-1]-un[1:-1,1:-1]*dt/dx*(un[1:-1,1:-1]-un[0:-2,1:-1])-\
		vn[1:-1,1:-1]*dt/dx*(un[1:-1,1:-1]-un[1:-1,0:-2])-\
		dt/(2*rho*dx)*(pn[2:,1:-1]-pn[0:-2,1:-1])+\
		nu*(dt/dx**2*(un[2:,1:-1]-2*un[1:-1,1:-1]+un[0:-2,1:-1]))+\
		nu*(dt/dy**2*(un[1:-1,2:]-2*un[1:-1,1:-1]+un[1:-1,0:-2]))
	
	v[1:-1,1:-1] = vn[1:-1,1:-1]-un[1:-1,1:-1]*dt/dx*(vn[1:-1,1:-1]-vn[0:-2,1:-1])-\
		vn[1:-1,1:-1]*dt/dx*(vn[1:-1,1:-1]-vn[1:-1,0:-2])-\
		dt/(2*rho*dx)*(pn[2:,1:-1]-pn[0:-2,1:-1])+\
		nu*(dt/dx**2*(vn[2:,1:-1]-2*vn[1:-1,1:-1]+vn[0:-2,1:-1]))+\
		nu*(dt/dy**2*(vn[1:-1,2:]-2*vn[1:-1,1:-1]+vn[1:-1,0:-2]))

	p[1:-1,1:-1] = ((pn[2:,1:-1]+pn[0:-2,1:-1])*dy**2+(pn[1:-1,2:]+pn[1:-1,0:-2])*dx**2)/\
			(2*(dx**2+dy**2)) -\
			rho*dx**2*dy**2/(2*(dx**2+dy**2))*\
			((1/dt*((un[2:,1:-2]-un[0:-2,1:-1])/(2*dx)+(vn[1:-1,2:]-vn[1:-1,0:-2])/(2*dy)))-\
			(un[2:,1:-1]-un[0:-2,1:-1])/(2*dx)*(un[2:,1:-1]-un[0:-2,1:-1])/(2*dx)-\	
			2*(un[1:-1,2:]-un[1:-1,0:-2])/(2*dy)*(vn[2:,1:-1]-vn[0:-2,1:-1])/(2*dx)-\
			(vn[1:-1,2:]-vn[1:-1,0:-2])/(2*dy)*(vn[1:-1,2:]-vn[1:-1,0:-2])/(2*dy)-\	
