##########2D Poisson##########
####This code converges on a solution to Poisson's equation in two dimensions using the finite difference method.
####
####Boundary Conditions:
####  p = 0 @ all borders
####  b = 0 everywhere except at (.5, .5) and (1.5, 1.5)
###############################
##Requires python >= 2.7, numpy and matplotlib
##############################
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy 

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
p = numpy.zeros((ny,nx)) ##create a XxY vector of 0's
pn = numpy.zeros((ny,nx)) ##create a XxY vector of 0's
b = numpy.zeros((ny,nx))

##plotting aids
x = numpy.linspace(0,2,nx)
y = numpy.linspace(0,1,ny)

##boundary conditions
p[:,0] = 0		##p = 0 @ x = 0
p[:,-1] = 0		##p = 0 @ x = 2
p[0,:] = 0		##p = 0 @ y = 0
p[-1,:] = 0		##p = 0 @ y = 1

b[numpy.round(nx/4.0),numpy.round(ny/4.0)] = 100
b[numpy.round(3*nx/4.0),numpy.round(3*ny/4.0)] = -100

##Initialize animation (plot ICs)
fig = plt.figure()
ax = fig.gca(projection='3d')
X,Y = numpy.meshgrid(x,y)


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

surf = ax.plot_surface(X,Y,p[:], rstride=1, cstride=1, cmap='PuOr',
        linewidth=0, antialiased=False)
#surf = ax.plot_wireframe(X,Y,p[:])
ax.set_xlim(0,2)
ax.set_ylim(0,1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.view_init(30,225)
plt.draw()
