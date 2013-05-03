import numpy
from 1d_nonlinear_numpy import *

nx = numpy.linspace(6,14,9)
nx = nx**2*1000

for i in nx:
    main(nx)
    print 'Completed run of %d elements' % (int(i))
