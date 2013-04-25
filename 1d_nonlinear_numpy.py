##1D linear convection
from pylab import *
import time

###Enter python interactive mode
ion()


###Variable declarations
nx = 81
nt = 100
c = 1
xmax = 4.0
dx = xmax/(nx-1)
sigma = 0.25
dt = sigma*dx

###Assign initial conditions
u = ones((1,nx)) ##create a 1xn vector of 1's
u[0,.5/dx:1/dx+1]=2 ##set hat function I.C. : .5<=x<=1 is 2
un = ones((1,nx))    


###Plot first frame of animation (ICs)
line, = plot(linspace(0,xmax,nx),u[0,:],marker='o',lw=2)


###Run through nt timesteps and plot/animate each step
for n in range(nt): ##loop across number of time steps
	un[:] = u[:]
	u[0,1:] = -un[0,1:]*dt/dx*(un[0,1:]-un[0,:-1])+un[0,1:]
	line.set_ydata(u[0,:])
	draw()
	title('1D Nonlinear Convection')  
  


