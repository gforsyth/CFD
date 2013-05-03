import numpy
import matplotlib.pyplot as plt


data = numpy.genfromtxt('numpycavitytimes')
datacud = numpy.genfromtxt('cudajit_cavity')

nx = data[:21:5]
cells = data[1:21:5]
runtime = data[2:21:5]
timestep = data[3:21:5]
iter = data[4:21:5]

nxc = datacud[:16:4]
runtimec = datacud[1:16:4]
dtc = datacud[2:16:4]
iterc = datacud[3:16:4]
cellsc = nxc**2

#
#plt.subplot(211)
#
#plt.plot(cells, runtime, '--')
#plt.plot(cellsc, runtimec, '-o')
#plt.xlabel('Number of cells')
#plt.ylabel('Runtime')
#plt.legend(['Numpy', 'Numba with CUDA'], loc=2)
#
#plt.subplot(212)
#
#plt.plot(iter, runtime, '--')
#plt.plot(iterc, runtimec, '-o')
#plt.xlabel('Time iterations')
#plt.ylabel('Runtime')
#plt.legend(['Numpy', 'Numba with CUDA'], loc=2)
#
#plt.savefig('numpyvsnumba_flow1.png')
#plt.show()

plt.plot(cells/iter, runtime, '--')
plt.plot(cellsc/iterc, runtimec, '-o')
plt.legend(['Numpy', 'Numba'])
plt.show()




data = numpy.genfromtxt('numpycavity0.1')
datacud = numpy.genfromtxt('cuda0.1')

nx = data[::5]
cells = data[1::5]
runtime = data[2::5]
timestep = data[3::5]
iter = data[4::5]

nxc = datacud[::4]
runtimec = datacud[1::4]
dtc = datacud[2::4]
iterc = datacud[3::4]
cellsc = nxc**2

plt.plot(cells/iter, runtime, '--')
plt.plot(cellsc/iterc, runtimec, '-o')
plt.legend(['Numpy', 'Numba'])
plt.show()
#
#plt.subplot(211)
#
#plt.plot(cells, runtime, '--')
#plt.plot(cellsc, runtimec, '-o')
#plt.xlabel('Number of cells')
#plt.ylabel('Runtime')
#plt.legend(['Numpy', 'Numba with CUDA'], loc=2)
#
#plt.subplot(212)
#
#plt.plot(iter, runtime, '--')
#plt.plot(iterc, runtimec, '-o')
#plt.xlabel('Time iterations')
#plt.ylabel('Runtime')
#plt.legend(['Numpy', 'Numba with CUDA'], loc=2)
#
#plt.savefig('numpyvsnumba_flow.1.png')
#plt.show()
