import matplotlib.pyplot as plt
import numpy

f = open('times','r')
i = 0
iter = []
num = []
vec = []
cud = []

for line in f:
    if i % 4 == 0:
        iter.append(line)
    elif i % 4 == 1:
        num.append(line)
    elif i % 4 == 2:
        vec.append(line)
    else:
        cud.append(line)
    i += 1

f.close()
x = len(cud)
anum = numpy.zeros(x)
avec = numpy.zeros(x)
acud = numpy.zeros(x)
aiter = numpy.zeros(x)


for i in range(x):
    anum[i] = float(num[i])
    avec[i] = float(vec[i])
    acud[i] = float(cud[i])
    aiter[i] = float(iter[i])
   
plt.plot(aiter[:], anum[:], '-o')
plt.plot(aiter[:], avec[:], '-o')
plt.plot(aiter[:], acud[:], '-o')
plt.legend(['Numpy', 'NumbaPro', 'NumbaPro with CUDA'], loc=2)
plt.title('Grid Size vs. Compute Time')
plt.xlabel('Number of elements')
plt.ylabel('Compute time in seconds')

plt.savefig('1dgridvtime.png')

plt.figure()
plt.plot(aiter[:5], anum[:5], '-o')
plt.plot(aiter[:5], avec[:5], '-o')
plt.plot(aiter[:5], acud[:5], '-o')
plt.legend(['Numpy', 'NumbaPro', 'NumbaPro with CUDA'], loc=2)
plt.title('Grid Size vs. Compute Time')
plt.xlabel('Number of elements')
plt.ylabel('Compute time in seconds')
plt.savefig('1dgridvtime_closeup.png')

