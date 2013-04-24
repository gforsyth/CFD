from numbapro import cuda, jit, float32
import numpy

@jit(argtypes=[float32[:], float32[:], float32[:]], target='gpu')
def cuda_sum(a, b, c):
        tid = cuda.threadIdx.x
        blkid = cuda.blockIdx.x
        blkdim = cuda.blockDim.x
        i = tid + blkid * blkdim
        c[i] = a[i] + b[i]

griddim = 10, 1
blockdim = 32, 1, 1
cuda_sum_configured = cuda_sum[griddim, blockdim]

a = numpy.array(numpy.random.random(320), dtype=numpy.float32)
b = numpy.array(numpy.random.random(320), dtype=numpy.float32)
c = numpy.empty_like(a)
cuda_sum_configured(a, b, c)

print c
