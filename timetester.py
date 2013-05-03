import numpy
from oned_nonlinear import *

nx = numpy.linspace(15,17,3)
nx = 2**nx*1000

for i in nx:
    main(i)
    print 'Completed run of %d elements' % (int(i))
