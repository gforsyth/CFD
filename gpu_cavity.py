import numpy
from numbapro import autojit, cuda, jit, float32
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import math
import time

################################################################################
################################################################################
################################################################################
### This code requires NumbaPro, available from Continuum Analytics (continuum.io) 
### and an NVIDIA GPU with Cuda 5.0 and a compute capability >= 2.0
################################################################################
################################################################################
################################################################################


###Decorator to tell NumbaPro to compile the following function into a CUDA kernel
@jit(argtypes=[float32[:,:], float32[:,:], float32[:,:], float32[:,:], float32[:,:], float32, float32, float32, float32, float32], target='gpu')
def CudaU(U, V, P, UN, VN, dx, dy, dt, rho, nu):

###Retrieve thread ID, block ID and block dimension for both x & y 
    tidx = cuda.threadIdx.x
    blkidx = cuda.blockIdx.x
    blkdimx = cuda.blockDim.x

    tidy = cuda.threadIdx.y
    blkidy = cuda.blockIdx.y
    blkdimy = cuda.blockDim.y

    m, n = U.shape
    
###Given thread ID, block ID and block dim, calculate corresponding i & j on grid
    i = tidx + blkidx * blkdimx             
    j = tidy + blkidy * blkdimy

    if i >= U.shape[0] or j >= U.shape[1]:
        return                              ###if you try to index out of the array bounds, kill the thread

####Calculate U velocity
    UN[i,j]=U[i,j]-U[i,j]*dt/dx*(U[i,j]-U[i-1,j])-\
        V[i,j]*dt/dy*(U[i,j]-U[i,j-1])-\
            dt/(2*rho*dx)*(P[i+1,j]-P[i-1,j])+\
        nu*(dt/dx**2*(U[i+1,j]-2*U[i,j]+U[i-1,j])+\
            dt/dy**2*(U[i,j+1]-2*U[i,j]+U[i,j-1]))

####Implement B.C. for U
    if i == 0:
        UN[i, j] = 0
    elif i == m-1:
        UN[i, j] = 0
    elif j == 0:
        UN[i, j] = 0
    elif j == n-1:
        UN[i, j] = 1

####Calculate V velocity
    VN[i,j]=V[i,j]-U[i,j]*dt/dx*(V[i,j]-V[i-1,j])-\
        V[i,j]*dt/dy*(V[i,j]-V[i,j-1])-\
            dt/(2*rho*dx)*(P[i,j+1]-P[i,j-1])+\
        nu*(dt/dx**2*(V[i+1,j]-2*V[i,j]+V[i-1,j])+\
            dt/dy**2*(V[i,j+1]-2*V[i,j]+V[i,j-1]))


####Implement B.C. for V
    if i == 0:
        VN[i, j] = 0
    elif i == m-1:
        VN[i, j] = 0
    elif j == 0:
        VN[i, j] = 0
    elif j == n-1:
        VN[i, j] = 0

####Copy updated values back to original U and V arrays before transfer off GPU
    U[i,j] = UN[i,j]
    V[i,j] = VN[i,j]


@autojit ###Target the following function for CPU vectorization (possible speed gains to be had by specifying variable inputs
def ppe(rho, dt, dx, dy, U, V, P):
    height, width = U.shape
    B = numpy.zeros((height, width))
    PN = numpy.zeros((height, width))
    nit = 50
    for i in range(1,width-1):
        for j in range(1, height-1):
            B[i,j] = 1/dt*((U[i+1,j]-U[i-1,j])/(2*dx)+\
                (V[i,j+1]-V[i,j-1])/(2*dy))-\
                    ((U[i+1,j]-U[i-1,j])/(2*dx))**2-\
                2*(U[i,j+1]-U[i,j-1])/(2*dy)*\
                    (V[i+1,j]-V[i-1,j])/(2*dx)-\
                    ((V[i,j+1]-V[i,j-1])/(2*dy))**2

    for n in range(nit):
        for i in range(1,width-1):
            for j in range(1, height-1):
                PN[i,j] = ((P[i+1,j]+P[i-1,j])*dy**2+\
                    (P[i,j+1] + P[i,j-1])*dx**2)/(2*(dx**2+dy**2))\
                        -rho*dx**2*dy**2/(2*(dx**2+dy**2))*B[i,j]

        for i in range(height):    
            PN[i, 0] = PN[i, 1]
            PN[i, width-1] = 0 
        for j in range(width):
            PN[0, j] = PN[1, j]
            PN[height-1,j] = PN[height-2, j]
    
        P[:] = PN[:]
    return P

def main():

    flowtime = 0.1
    nx = 128 
    ny = 128
    dx = 2.0/(nx-1)
    dy = 2.0/(ny-1)

    dt = dx/50 ##ensures stability for a given mesh fineness
    
    rho = 1.0
    nu =.1 

    nt = int(flowtime/dt) ##calculate number of timesteps required to reach a specified total flowtime

    U = numpy.zeros((nx,ny), dtype=numpy.float32)
    U[-1,:] = 1
    V = numpy.zeros((nx,ny), dtype=numpy.float32)
    P = numpy.zeros((ny, nx), dtype=numpy.float32)
    UN = numpy.zeros((nx,ny), dtype=numpy.float32)
    VN = numpy.zeros((nx,ny), dtype=numpy.float32)

    griddim = nx, ny
    blockdim = 768, 768, 1
    #if nx > 767:
    #    griddim = int(math.ceil(float(nx)/blockdim[0])), int(math.ceil(float(ny)/blockdim[0]))

    t1 = time.time()    
    ###Target the GPU to begin calculation
    stream = cuda.stream()
    d_U = cuda.to_device(U, stream)
    d_V = cuda.to_device(V, stream)
    d_UN = cuda.to_device(UN, stream)
    d_VN = cuda.to_device(VN, stream)

    for i in range(nt):
        P = ppe(rho, dt, dx, dy, U, V, P)
        CudaU[griddim, blockdim, stream](d_U, d_V, P, d_UN, d_VN, dx, dy, dt, rho, nu)
        d_U.to_host(stream)
        d_V.to_host(stream)
        stream.synchronize()

    t2 = time.time()

    print "Completed grid of %d by %d in %.6f seconds" % (nx, ny, t2-t1)
    x = numpy.linspace(0,2,nx)
    y = numpy.linspace(0,2,ny)
    Y,X = numpy.meshgrid(y,x)
    
    #prescon = plt.figure()
    #plt.contourf(X[::10,::10],Y[::10,::10],P[::10,::10],alpha=0.5)
    #plt.colorbar()
    #plt.contour(X[::10,::10],Y[::10,::10],P[::10,::10])
    #plt.quiver(X[::10,::10],Y[::10,::10],U[::10,::10],V[::10,::10])
    #plt.xlabel('X')
    #plt.ylabel('Y')
    #plt.title('Pressure contour')
    #plt.contourf(X,Y,P,alpha=.5)
    #plt.colorbar()
    #plt.contour(X,Y,P,)
    #plt.quiver(X[::2,::2],Y[::2,::2],U[::2,::2],V[::2,::2])
    #plt.xlabel('X')
    #plt.ylabel('Y')
    #plt.title('Pressure contour')
    
    #plt.show()
    #f = open('cudajit_cavity', 'a')
    #f.write(str(nx)+'\n')
    #f.write(str(t2-t1) +'\n')
    #f.write(str(dt) + '\n')
    #f.write(str(nt) + '\n')
    #f.close()

   # from ghiacompy import plotghiacomp

   # plotghiacomp(U[(nx-1)/2,:],numpy.linspace(0,1,ny))

if __name__ == "__main__":
        main()
