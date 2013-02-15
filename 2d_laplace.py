##2D Laplace
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import os

plt.ion()

##variable declarations
nx = 31
ny = 31
c = 1
dx = 2.0/(nx-1)
dy = 2.0/(ny-1)


##initial conditions
p = np.zeros((ny,nx)) ##create a XxY vector of 0's
pn = np.zeros((ny,nx)) ##create a XxY vector of 0's


##plotting aids
x = np.linspace(0,2,nx)
y = np.linspace(0,1,ny)

##boundary conditions
p[:,0] = 0		##p = 0 @ x = 0
p[:,-1] = y		##p = y @ x = 2
p[0,:] = p[1,:]		##dp/dy = 0 @ y = 0
p[-1,:] = p[-2,:]	##dp/dy = 0 @ y = 1

##Initialize animation (plot ICs)
fig = plt.figure()
ax = fig.gca(projection='3d')
X,Y = np.meshgrid(x,y)
surf = ax.plot_surface(X,Y,p[:], rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
ax.set_xlim(0,2)
ax.set_ylim(0,1)
ax.view_init(30,225)
plt.draw()

l1norm = 1
n = 1

##time loop
while l1norm > .0001:
	pn[:] = p[:]
	p[1:-1,1:-1] = (dy**2*(pn[2:,1:-1]+pn[0:-2,1:-1])+dx**2*(pn[1:-1,2:]+pn[1:-1,0:-2]))/(2*(dx**2+dy**2)) 
	p[0,0] = (dy**2*(pn[1,0]+pn[-1,0])+dx**2*(pn[0,1]+pn[0,-1]))/(2*(dx**2+dy**2))
	p[-1,-1] = (dy**2*(pn[0,-1]+pn[-2,-1])+dx**2*(pn[-1,0]+pn[-1,-2]))/(2*(dx**2+dy**2)) 

	p[:,0] = 0		##p = 0 @ x = 0
	p[:,-1] = y		##p = y @ x = 2
	p[0,:] = p[1,:]		##dp/dy = 0 @ y = 0
	p[-1,:] = p[-2,:]	##dp/dy = 0 @ y = 1
	l1norm = (np.sum(np.abs(p[:])-np.abs(pn[:])))/np.sum(np.abs(pn[:]))
	print l1norm
	n = n+1
	if n%40 == 0:
		surf.remove()
		surf = ax.plot_surface(X,Y,p[:], rstride=1, cstride=1, cmap=cm.coolwarm,
			linewidth=0, antialiased=False)
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.view_init(30,225)
		plt.draw()
surf.remove()
surf = ax.plot_surface(X,Y,p[:], rstride=1, cstride=1, cmap=cm.coolwarm,
	linewidth=0, antialiased=False)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.view_init(30,225)
