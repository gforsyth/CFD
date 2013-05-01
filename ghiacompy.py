import numpy
import matplotlib.pyplot as plt

def plotghiacomp(U, Y):
    Ghia_Y = numpy.array([0.0,0.0547,.0625,.0703,.1016,.1719,.2813,.4531,.5,.6172,.7344,.8516,.9531,.9609,.9688,.9766,1.0])
    Re100 = numpy.array([0.0000,-.03717,-.04192,-.04775,-.06434,-.10150,-.15662,\
                    -.2109,-.20581,-.13641,.00332,.23151,.68717,.73722,.78871,.84123,1.0])

    plt.plot(Re100, Ghia_Y, 'o')

    plt.plot(U, Y)
    plt.show()
