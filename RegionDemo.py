import random

import matplotlib
import AtomicRegion as ar
import numpy as np

#
# regionA = reg.Region(100)
# regionB = reg.Region(100)
# regionC = reg.Region(100)
# regionA.join_cross(regionB.output_v)
# regionB.join_cross(regionC.output_v)
# regionC.join_cross(regionA.output_v)
# regionA.iterate()
# regionB.iterate()
# regionC.iterate()
# regionA.iterate()
# print(regionA.output_v)
np.set_printoptions(linewidth=2000)
regionA = ar.AtomicRegion(100)
regionB = ar.AtomicRegion(100)
regionB.join_cross(regionA.cross_v, 'random')
regionA.join_cross(regionA.cross_v, 'random')



for j in range(500):
    regionA.output_v[0, 0] = random.random()
    regionA.output_v[2, 0] = random.random()
    regionA.iterate()
    regionA.output_v[0, 0] = random.random()
    regionA.output_v[2, 0] = random.random()
    regionA.iterate()
    regionB.iterate()
    print(np.transpose(regionA.output_v))
for i in range(500):
    regionA.output_v[0, 0] = random.random()
    regionA.output_v[2, 0] = random.random()
    regionA.iterate()
    regionA.output_v[0, 0] = random.random()
    regionA.output_v[2, 0] = random.random()
    regionA.iterate()
    regionB.iterate()
    print(np.transpose(regionA.output_v))
for k in range(10000):
    regionA.output_v[0, 0] = 1
    regionA.output_v[2, 0] = 1
    regionA.iterate()
    regionA.output_v[0, 0] = 0
    regionA.output_v[2, 0] = 0
    regionA.iterate()
    regionB.iterate()
    print(np.transpose(regionB.output_v))
