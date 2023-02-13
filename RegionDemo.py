import region as reg

regionA = reg.Region(100)
regionB = reg.Region(100)
regionB.join_cross(regionA.output_v)
print(regionA.output_v)
regionB.iterate()
print(regionA.output_v)
