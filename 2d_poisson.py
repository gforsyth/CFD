##2D Laplace
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import os

plt.ion()

##variable declarations
nx = 81
ny = 81
nt = 300
c = 1
dx = 2.0/(nx-1)
dy = 2.0/(ny-1)
sigma = .5
dt = sigma*dx*dy


##initial conditions
p = np.zeros((ny,nx)) ##create a XxY vector of 0's
pn = np.zeros((ny,nx)) ##create a XxY vector of 0's
b = np.zeros((ny,nx))

##plotting aids
x = np.linspace(0,2,nx)
y = np.linspace(0,1,ny)

##boundary conditions
p[:,0] = 0		##p = 0 @ x = 0
p[:,-1] = 0		##p = 0 @ x = 2
p[0,:] = 0		##p = 0 @ y = 0
p[-1,:] = 0		##p = 0 @ y = 1

b[np.round(nx/4.0),np.round(ny/4.0)] = 100
b[np.round(3*nx/4.0),np.round(3*ny/4.0)] = -100

##Initialize animation (plot ICs)
fig = plt.figure()
ax = fig.gca(projection='3d')
X,Y = np.meshgrid(x,y)


##time loop
for n in range(nt):
	pn[:] = p[:]
	p[1:-1,1:-1] = (dy**2*(pn[2:,1:-1]+pn[0:-2,1:-1])+dx**2*(pn[1:-1,2:]+pn[1:-1,0:-2])-b[1:-1,1:-1]*dx**2*dy**2)/(2*(dx**2+dy**2)) 
	p[0,0] = (dy**2*(pn[1,0]+pn[-1,0])+dx**2*(pn[0,1]+pn[0,-1])-b[0,0]*dx**2*dy**2)/(2*(dx**2+dy**2))
	p[-1,-1] = (dy**2*(pn[0,-1]+pn[-2,-1])+dx**2*(pn[-1,0]+pn[-1,-2])-b[-1,-1]*dx**2*dy**2)/(2*(dx**2+dy**2)) 

	p[:,0] = 0		##p = 0 @ x = 0
	p[:,-1] = 0		##p = 0 @ x = 2
	p[0,:] = 0		##p = 0 @ y = 0
	p[-1,:] = 0		##p = 0 @ y = 1
	#if n%20 == 0:
	#surf.remove()
	#surf = ax.plot_surface(X,Y,p[:], rstride=1, cstride=1, cmap=cm.coolwarm,
	#	linewidth=0, antialiased=False)
	#surf = ax.plot_wireframe(X,Y,p[:])

surf = ax.plot_surface(X,Y,p[:], rstride=1, cstride=1, cmap='PuOr',
        linewidth=0, antialiased=False)
surf = ax.plot_wireframe(X,Y,p[:])
ax.set_xlim(0,2)
ax.set_ylim(0,1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.view_init(30,225)
plt.draw()
