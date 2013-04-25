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
   
plt.plot(aiter[:], anum[:])
plt.plot(aiter[:], avec[:])
plt.plot(aiter[:], acud[:])
plt.show()
