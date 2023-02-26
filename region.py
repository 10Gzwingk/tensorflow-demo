
class Region:
    def join_cross(self, v, join_type):
        raise NotImplementedError

    def exit_cross(self, v):
        raise NotImplementedError

    # 网络迭代
    def iterate(self):
        raise NotImplementedError

    # 更新权重
    def update(self, x, v):
        raise NotImplementedError
