import numpy
from numbapro import autojit, cuda, jit, float32



@jit(argtypes=[float32[:,:], float32[:,:], float32[:,:], float32, float32, float32, float32], target='gpu')
def CudaU(U, V, P, dx, dy, dt, rho):
    tid = cuda.threadIdx.x
    blkid = cuda.blockIdx.x
    blkdim = cuda.blockDim.x
    ntid = tid + blkid * blkdim

    height, width = U.shape

    i = ntid % width
    j = ntid / width

    UN[i,j]=U[i,j]-U[i,j]*dt/dx*(U[i,j]-U[i-1,j])-\
        V[i,j]*dt/dy*(U[i,j]-U[i,j-1])-\
            dt/(2*rho*dx)*(P[i+1,j]-P[i-1,j])+\
        nu*(dt/dx**2*(U[i+1,j]-2*U[i,j]+U[i-1,j])+\
            dt/dy**2*(U[i,j+1]-2*U[i,j]+U[i,j-1]))


@autojit
def ppe(rho, dt, dx, dy, U, V, P):
	height, width = U.shape
    B = np.zeros((height, width))
    PN = np.zeros((height, width))

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

        P[:] = PN[:]


def main():
    

if __name__ == "__main__":
        main()
