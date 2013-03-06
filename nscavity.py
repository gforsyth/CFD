##Cavity Flow Navier Stokes
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import os

plt.ion()

anicheck = raw_input('Do you want to animate the results?(y/n): ')

##variable declarations
nx = 41
ny = 41
nt = 500
nit=50
c = 1
dx = 2.0/(nx-1)
dy = 2.0/(ny-1)
x = np.linspace(0,2,nx)
y = np.linspace(0,2,ny)
Y,X = np.meshgrid(y,x)


##physical variables
rho = 1
nu = .5

dt = input('Enter dt: ') 

u = np.zeros((ny,nx)) ##create a XxY vector of 0's
un = np.zeros((ny,nx)) ##create a XxY vector of 0's

v = np.zeros((ny,nx)) ##create a XxY vector of 0's
vn = np.zeros((ny,nx)) ##create a XxY vector of 0's

p = np.zeros((ny,nx)) ##create a XxY vector of 0's
pn = np.zeros((ny,nx)) ##create a XxY vector of 0's

b = np.zeros((ny,nx))

#f, (ax1, ax2) = plt.subplots(ncols=2)
plt.figure()
for n in range(nt):
	un[:] = u[:]
	vn[:] = v[:]

	b[1:-1,1:-1]=rho*(1/dt*((u[2:,1:-1]-u[0:-2,1:-1])/(2*dx)+(v[1:-1,2:]-v[1:-1,0:-2])/(2*dy))-\
		((u[2:,1:-1]-u[0:-2,1:-1])/(2*dx))**2-\
		2*((u[1:-1,2:]-u[1:-1,0:-2])/(2*dy)*(v[2:,1:-1]-v[0:-2,1:-1])/(2*dx))-\
		((v[1:-1,2:]-v[1:-1,0:-2])/(2*dy))**2)	

	for q in range(nit):
		pn[:] = p[:]
		p[1:-1,1:-1] = ((pn[2:,1:-1]+pn[0:-2,1:-1])*dy**2+(pn[1:-1,2:]+pn[1:-1,0:-2])*dx**2)/\
			(2*(dx**2+dy**2)) -\
			dx**2*dy**2/(2*(dx**2+dy**2))*b[1:-1,1:-1]
		
		p[-1,:] =p[-2,:]		##p = 0 at y = 2
		p[0,:] = p[1,:]		##dp/dy = 0 at y = 0
		p[:,0]=p[:,1]		##dp/dx = 0 at x = 0
		p[:,-1]=0		##dp/dx = 0 at x = 2
		

	
	u[1:-1,1:-1] = un[1:-1,1:-1]-\
		un[1:-1,1:-1]*dt/dx*(un[1:-1,1:-1]-un[0:-2,1:-1])-\
		vn[1:-1,1:-1]*dt/dy*(un[1:-1,1:-1]-un[1:-1,0:-2])-\
		dt/(2*rho*dx)*(p[2:,1:-1]-p[0:-2,1:-1])+\
		nu*(dt/dx**2*(un[2:,1:-1]-2*un[1:-1,1:-1]+un[0:-2,1:-1])+\
		dt/dy**2*(un[1:-1,2:]-2*un[1:-1,1:-1]+un[1:-1,0:-2]))
	
	v[1:-1,1:-1] = vn[1:-1,1:-1]-\
		un[1:-1,1:-1]*dt/dx*(vn[1:-1,1:-1]-vn[0:-2,1:-1])-\
		vn[1:-1,1:-1]*dt/dy*(vn[1:-1,1:-1]-vn[1:-1,0:-2])-\
		dt/(2*rho*dy)*(p[1:-1,2:]-p[1:-1,0:-2])+\
		nu*(dt/dx**2*(vn[2:,1:-1]-2*vn[1:-1,1:-1]+vn[0:-2,1:-1])+\
		(dt/dy**2*(vn[1:-1,2:]-2*vn[1:-1,1:-1]+vn[1:-1,0:-2])))

	u[0,:] = 0
	u[:,0] = 0
	u[:,-1] = 1
	v[0,:] = 0
	v[-1,:]=0
	v[:,0] = 0
	v[:,-1] = 0
	u[-1,:] = 0		## at y = 2 where u = 1
	

	if anicheck=='y' and n%5==0:
		plt.clf()
		plt.contourf(X,Y,p)
		#plt.quiver(X[::2,::2],Y[::2,::2],u[::2,::2],v[::2,::2])
		plt.quiver(X,Y,u,v)
		plt.draw()
	if anicheck=='n':
		plt.close()

#prescon = plt.figure()
#plt.contourf(X,Y,p,alpha=0.5)
#plt.colorbar()
#plt.contour(X,Y,p)
#plt.quiver(X[::2,::2],Y[::2,::2],u[::2,::2],v[::2,::2])
#plt.xlabel('X')
#plt.ylabel('Y')
#plt.title('Pressure contour')
#prescon.savefig('cavity pressure contour.png')
#
#stream = plt.figure()
#plt.streamplot(x,y,np.transpose(u),np.transpose(v),density=[5, 2])
#plt.contourf(X,Y,p,alpha=0.5)
#plt.colorbar()
#plt.xlabel('X')
#plt.ylabel('Y')
#plt.title('Velocity field streamplot')
#stream.savefig('cavity streamplot.png')
