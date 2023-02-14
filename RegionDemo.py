import region as reg
import numpy as np

#
regionA = reg.Region(100)
regionB = reg.Region(100)
regionC = reg.Region(100)
regionA.join_cross(regionB.output_v)
regionB.join_cross(regionC.output_v)
regionC.join_cross(regionA.output_v)
regionA.iterate()
regionB.iterate()
regionC.iterate()
regionA.iterate()
print(regionA.output_v)

