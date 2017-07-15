import numpy
import sys
a = numpy.load(sys.argv[1])
numpy.savetxt(sys.argv[1]+".csv", a, delimiter=",")
