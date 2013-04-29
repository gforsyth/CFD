import numpy
from numbapro import autojit, cuda, jit, float32



@jit(argtypes=[float32[:,:], float32[:,:], float32[:,:], float32, float32, float32, float32, float32, float32[:,:], float32[:,:]], target='gpu')
def CudaU(U, V, P, dx, dy, dt, rho, nu, UN, VN):
    tidx = cuda.threadIdx.x
    blkidx = cuda.blockIdx.x
    blkdimx = cuda.blockDim.x

    tidy = cuda.threadIdx.y
    blkidy = cuda.blockIdx.y
    blkdimy = cuda.blockDim.y

    height, width = U.shape

    i = tidx + blkidx * blkdimx
    j = tidy + blkidy * blkdimy

    if i >= U.shape[0] or j >= U.shape[1]:
        return                              ###if you try to index out of the array bounds, kill the thread


    UN[i,j]=U[i,j]-U[i,j]*dt/dx*(U[i,j]-U[i-1,j])-\
        V[i,j]*dt/dy*(U[i,j]-U[i,j-1])-\
            dt/(2*rho*dx)*(P[i+1,j]-P[i-1,j])+\
        nu*(dt/dx**2*(U[i+1,j]-2*U[i,j]+U[i-1,j])+\
            dt/dy**2*(U[i,j+1]-2*U[i,j]+U[i,j-1]))

    if i == 0:
        UN[i, j] = 0
    elif i == width-1:
        UN[i, j] = 0
    elif j == 0:
        UN[i, j] = 0
    elif j == height-1:
        UN[i, j] = 1

    VN[i,j]=V[i,j]-U[i,j]*dt/dx*(V[i,j]-V[i-1,j])-\
        V[i,j]*dt/dy*(V[i,j]-V[i,j-1])-\
            dt/(2*rho*dx)*(P[i,j+1]-P[i,j-1])+\
        nu*(dt/dx**2*(V[i+1,j]-2*V[i,j]+V[i-1,j])+\
            dt/dy**2*(V[i,j+1]-2*V[i,j]+V[i,j-1]))

    if i == 0:
        VN[i, j] = 0
    elif i == width-1:
        VN[i, j] = 0
    elif j == 0:
        VN[i, j] = 0
    elif j == height-1:
        VN[i, j] = 0

    U[i,j] = UN[i,j]
    V[i,j] = VN[i,j]


#@jit(argtypes=[float32[:,:], float32[:,:], float32[:,:], float32, float32, float32, float32, float32, float32[:,:]], target='gpu')
#def CudaV(U, V, P, dx, dy, dt, rho, nu, VN):
#    tidx = cuda.threadIdx.x
#    blkidx = cuda.blockIdx.x
#    blkdimx = cuda.blockDim.x
#
#    tidy = cuda.threadIdx.y
#    blkidy = cuda.blockIdx.y
#    blkdimy = cuda.blockDim.y
#
#    height, width = U.shape
#
#    i = tidx + blkidx * blkdimx
#    j = tidy + blkidy * blkdimy
#
#    VN[i,j]=V[i,j]-U[i,j]*dt/dx*(V[i,j]-V[i-1,j])-\
#        V[i,j]*dt/dy*(V[i,j]-V[i,j-1])-\
#            dt/(2*rho*dx)*(P[i,j+1]-P[i,j-1])+\
#        nu*(dt/dx**2*(V[i+1,j]-2*V[i,j]+V[i-1,j])+\
#            dt/dy**2*(V[i,j+1]-2*V[i,j]+V[i,j-1]))
#
#    if i == 0:
#        VN[i, j] = 0
#    elif i == width-1:
#        VN[i, j] = 0
#    elif j == 0:
#        VN[i, j] = 0
#    elif j == height-1:
#        VN[i, j] = 0
#
@autojit
def ppe(rho, dt, dx, dy, U, V, P):
    height, width = U.shape
    B = numpy.zeros((height, width))
    PN = numpy.zeros((height, width))
    nit = 50
    for i in range(1,width):
        for j in range(1, height):
            B[i,j] = 1/dt*((U[i+1,j]-U[i-1,j])/(2*dx)+(V[i,j+1]-V[i,j-1])/(2*dy))\
                    -((U[i+1,j]-U[i-1,j])/(2*dx))**2\
                    -2*(U[i,j+1]-U[i,j-1])/(2*dy)*(V[i+1,j]-V[i-1,j])/(2*dx)\
                    -((V[i,j+1]-V[i,j-1])/(2*dy))**2

    for n in range(nit):
        for i in range(1,width):
            for j in range(1, height):
                PN[i,j] = ((P[i+1,j]+P[i-1,j])*dy**2+(P[i,j+1] + P[i,j-1])*dx**2)/(2*(dx**2+dy**2))\
                        +rho*dx**2*dy**2/((2*(dx**2+dy**2)))*B[i,j]

        for i in range(width):    
            PN[i, 0] = PN[i, 1]
            PN[i, height-1] = PN[i, height-2]
        for j in range(height):
            PN[0, j] = PN[1, j]
            PN[width-1,j] = PN[width-2, j]

        P[:] = PN[:]
    return P

def main():


    nx = 41
    ny = 41
    dx = 2.0/(nx-1)
    dy = 2.0/(ny-1)
    dt = .001

    rho = 1
    nu =.1 

    nt = 300

    U = numpy.zeros((ny,nx), dtype=numpy.float32)
    V = numpy.zeros((ny,nx), dtype=numpy.float32)
    P = numpy.zeros((ny, nx), dtype=numpy.float32)

    UN = numpy.zeros((ny,nx), dtype=numpy.float32)
    VN = numpy.zeros((ny,nx), dtype=numpy.float32)

    
    P = ppe(rho, dt, dx, dy, U, V, P)


    griddim = 1, 1
    blockdim = 64, 64, 1
    
    stream = cuda.stream()
    d_U = cuda.to_device(U, stream)
    d_V = cuda.to_device(V, stream)
    d_UN = cuda.to_device(UN, stream)
    d_VN = cuda.to_device(VN, stream)
   # d_P = cuda.to_device(P, stream)


#    CudaV_conf = CudaV[griddim, blockdim]
#    CudaU_conf = CudaU[griddim, blockdim]


    for n in range(nt):
        CudaU[griddim, blockdim, stream](d_U, d_V, P, dx, dy, dt, rho, nu, d_UN, d_VN)
        d_U.to_host(stream)
        d_V.to_host(stream)
        stream.synchronize()
        P = ppe(rho, dt, dx, dy, U, V, P)


if __name__ == "__main__":
        main()
