##Cavity Flow Navier Stokes
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import os

plt.ion()

##variable declarations
nx = 41
ny = 41
nt =300  
c = 1
dx = 2.0/(nx-1)
dy = 2.0/(ny-1)
X = np.linspace(0,2,nx)
Y = np.linspace(0,2,ny)


##physical variables
rho = .2
nu = .2

dt = dx*dy*nu*rho
##initial conditions
u = np.zeros((ny,nx)) ##create a XxY vector of 0's
un = np.zeros((ny,nx)) ##create a XxY vector of 0's

v = np.zeros((ny,nx)) ##create a XxY vector of 0's
vn = np.zeros((ny,nx)) ##create a XxY vector of 0's

p = np.zeros((ny,nx)) ##create a XxY vector of 0's
pn = np.zeros((ny,nx)) ##create a XxY vector of 0's



myplot = plt.streamplot(X,Y,u,v)
plt.show()

for n in range(nt):
	un[:] = u[:]
	vn[:] = v[:]
	pn[:] = p[:]

	u[1:-1,1:-1] = un[1:-1,1:-1]-un[1:-1,1:-1]*dt/dx*(un[1:-1,1:-1]-un[0:-2,1:-1])-\
		vn[1:-1,1:-1]*dt/dx*(un[1:-1,1:-1]-un[1:-1,0:-2])-\
		dt/(2*rho*dx)*(pn[2:,1:-1]-pn[0:-2,1:-1])+\
		nu*(dt/dx**2*(un[2:,1:-1]-2*un[1:-1,1:-1]+un[0:-2,1:-1])+\
		dt/dy**2*(un[1:-1,2:]-2*un[1:-1,1:-1]+un[1:-1,0:-2]))
	
	v[1:-1,1:-1] = vn[1:-1,1:-1]-un[1:-1,1:-1]*dt/dx*(vn[1:-1,1:-1]-vn[0:-2,1:-1])-\
		vn[1:-1,1:-1]*dt/dx*(vn[1:-1,1:-1]-vn[1:-1,0:-2])-\
		dt/(2*rho*dy)*(pn[1:-1,2:]-pn[1:-1,0:-2])+\
		nu*(dt/dx**2*(vn[2:,1:-1]-2*vn[1:-1,1:-1]+vn[0:-2,1:-1])+\
		(dt/dy**2*(vn[1:-1,2:]-2*vn[1:-1,1:-1]+vn[1:-1,0:-2])))

	p[1:-1,1:-1] = ((pn[2:,1:-1]+pn[0:-2,1:-1])*dy**2+(pn[1:-1,2:]+pn[1:-1,0:-2])*dx**2)/\
		(2*(dx**2+dy**2)) -\
		rho*dx**2*dy**2/(2*(dx**2+dy**2))*\
		((1/dt*((un[2:,1:-1]-un[0:-2,1:-1])/(2*dx)+(vn[1:-1,2:]-vn[1:-1,0:-2])/(2*dy)))-\
		(un[2:,1:-1]-un[0:-2,1:-1])/(2*dx)*(un[2:,1:-1]-un[0:-2,1:-1])/(2*dx)-\
		2*(un[1:-1,2:]-un[1:-1,0:-2])/(2*dy)*(vn[2:,1:-1]-vn[0:-2,1:-1])/(2*dx)-\
		(vn[1:-1,2:]-vn[1:-1,0:-2])/(2*dy)*(vn[1:-1,2:]-vn[1:-1,0:-2])/(2*dy))
###
###7	.	.	.	.	.
###6	.	.	.	.	.
###5	.	.	.	.	.
###4	.	.	.	.	.
###3	.	.	.	.	.
###2	.	.	.	.	.
###1	.	.	.	.	.
###0	.	.	.	.	.
###     0	1	2	3	4

	u[0,:] = 0
	u[:,0] = 0
	u[:,-1] = 0
	v[0,:] = 0
	v[-1,:]=0
	v[:,0] = 0
	v[:,-1] = 0
	
	u[-1,:] = 1		## at y = 2 where u = 1
	p[-1,:] = 0		##p = 0 at y = 2
	p[0,:] = p[1,:]		##dp/dy = 0 at y = 0
	p[:,0]=p[:,1]		##dp/dx = 0 at x = 0
	p[:,-1]=p[:,-2]		##dp/dx = 0 at x = 2
	if n%25 == 0:
		myplot.remove()
		myplot = plt.streamplot(X,Y,u,v)
		plt.show()
		wait = raw_input('enter to continue')
