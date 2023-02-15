import region as reg
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
regionA = reg.Region(100)
for i in range(500):
    regionA.iterate()
    regionA.output_v[0, 0] = 1
    print(np.transpose(regionA.output_v))
for k in range(500):
    regionA.iterate()
    print(np.transpose(regionA.output_v))

