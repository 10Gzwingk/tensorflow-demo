
class Region:
    def join_cross(self, v, join_type):
        raise NotImplementedError

    def exit_cross(self, v):
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError

    def update(self, x, v):
        raise NotImplementedError
