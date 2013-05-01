import numpy
from numbapro import autojit, cuda, jit, float32
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import math
import time


@jit(argtypes=[float32[:,:], float32[:,:], float32[:,:], float32, float32, float32, float32, float32, float32], target='gpu')
def CudaU(U, V, P, dx, dy, dt, rho, nu, nt):
    tidx = cuda.threadIdx.x
    blkidx = cuda.blockIdx.x
    blkdimx = cuda.blockDim.x

    tidy = cuda.threadIdx.y
    blkidy = cuda.blockIdx.y
    blkdimy = cuda.blockDim.y

    m, n = U.shape

    UN = cuda.shared.array(shape=(m, n), dtype=float32)
    VN = cuda.shared.array(shape=(m, n), dtype=float32)
    
    i = tidx + blkidx * blkdimx
    j = tidy + blkidy * blkdimy

    if i >= U.shape[0] or j >= U.shape[1]:
        return                              ###if you try to index out of the array bounds, kill the thread

    for n in range(nt):
        P = ppe(rho, dt, dx, dy, U, V, P)

        UN[i,j]=U[i,j]-U[i,j]*dt/dx*(U[i,j]-U[i-1,j])-\
            V[i,j]*dt/dy*(U[i,j]-U[i,j-1])-\
                dt/(2*rho*dx)*(P[i+1,j]-P[i-1,j])+\
            nu*(dt/dx**2*(U[i+1,j]-2*U[i,j]+U[i-1,j])+\
                dt/dy**2*(U[i,j+1]-2*U[i,j]+U[i,j-1]))

        if i == 0:
            UN[i, j] = 0
        elif i == m-1:
            UN[i, j] = 0
        elif j == 0:
            UN[i, j] = 0
        elif j == n-1:
            UN[i, j] = 1

        cuda.syncthreads()


        VN[i,j]=V[i,j]-U[i,j]*dt/dx*(V[i,j]-V[i-1,j])-\
            V[i,j]*dt/dy*(V[i,j]-V[i,j-1])-\
                dt/(2*rho*dx)*(P[i,j+1]-P[i,j-1])+\
            nu*(dt/dx**2*(V[i+1,j]-2*V[i,j]+V[i-1,j])+\
                dt/dy**2*(V[i,j+1]-2*V[i,j]+V[i,j-1]))

        if i == 0:
            VN[i, j] = 0
        elif i == m-1:
            VN[i, j] = 0
        elif j == 0:
            VN[i, j] = 0
        elif j == n-1:
            VN[i, j] = 0


        U[i,j] = UN[i,j]
        V[i,j] = VN[i,j]


        cuda.syncthreads()


@autojit
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

    flowtime = 1.0
    nx = 257
    ny = 257
    dx = 2.0/(nx-1)
    dy = 2.0/(ny-1)

    dt = dx/50
    
    rho = 1.0
    nu =.1 

    nt = int(flowtime/dt)

    U = numpy.zeros((nx,ny), dtype=numpy.float32)
    U[-1,:] = 1
    V = numpy.zeros((nx,ny), dtype=numpy.float32)
    P = numpy.zeros((ny, nx), dtype=numpy.float32)

    griddim = nx, ny
    blockdim = 768, 1, 1
    if nx > 767:
        griddim = int(math.ceil(float(nx)/blockdim[0])), int(math.ceil(float(ny)/blockdim[0]))

    t1 = time.time()    
    stream = cuda.stream()
    d_U = cuda.to_device(U, stream)
    d_V = cuda.to_device(V, stream)


    CudaU[griddim, blockdim, stream](d_U, d_V, P, dx, dy, dt, rho, nu, nt)
    d_U.to_host(stream)
    d_V.to_host(stream)
    stream.synchronize()

    t2 = time.time()

    print "Completed grid of %d by %d in %.6f seconds" % (nx, ny, t2-t1)
    x = numpy.linspace(0,2,nx)
    y = numpy.linspace(0,2,ny)
    Y,X = numpy.meshgrid(y,x)
    
    prescon = plt.figure()
    #plt.contourf(X[::10,::10],Y[::10,::10],P[::10,::10],alpha=0.5)
    #plt.colorbar()
    #plt.contour(X[::10,::10],Y[::10,::10],P[::10,::10])
    #plt.quiver(X[::10,::10],Y[::10,::10],U[::10,::10],V[::10,::10])
    #plt.xlabel('X')
    #plt.ylabel('Y')
    #plt.title('Pressure contour')
    plt.contourf(X,Y,P,alpha=.5)
    plt.colorbar()
    plt.contour(X,Y,P,)
    plt.quiver(X[::2,::2],Y[::2,::2],U[::2,::2],V[::2,::2])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Pressure contour')
    
    plt.show()

    from ghiacompy import plotghiacomp

    plotghiacomp(U[(nx-1)/2,:],numpy.linspace(0,1,ny))

if __name__ == "__main__":
        main()
