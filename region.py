
class Region:
    def connect(self, region, connect_params):
        raise NotImplementedError

    def disconnect(self, region):
        raise NotImplementedError

    def join_cross(self, v, join_type):
        raise NotImplementedError

    # 网络迭代
    def iterate(self):
        raise NotImplementedError

    # 更新权重
    def update(self, x, v):
        raise NotImplementedError

    def active(self, params):
        raise NotImplementedError
