class RegionScheduler:
    def __init__(self, regions):
        self.regions = regions

    def add(self, region):
        self.regions.append(region)

    def schedule(self):
        for region in self.regions:
            region.iterate()
