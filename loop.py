import numpy as numpy
numpy.set_printoptions(linewidth=400)

met = (numpy.random.random((400, 400)) - 0.5) * 2
v = numpy.random.random((400, 1))
for i in range(1000):
    v = numpy.minimum(numpy.maximum(numpy.matmul(met, v), 0), 1)
    v[0, 0] = 1
    print(numpy.reshape(v, newshape=(1, 400))[0:400])