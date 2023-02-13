import numpy as np

one = np.ones([1, 1])


def random(size):
    return (np.random.random(size) - 0.5) * 2


class Region:

    def __init__(self, size):
        self.cross_list = []
        self.size = size
        self.fc_net = random((size, size))  # 当前 region 的全连接矩阵
        self.output_v = random((size, 1))   # 当前 region 的神经元输出向量
        self.cross_net = random((size, 1))  # 交叉矩阵第一列代表偏移
        self.offset_v = random((size, 1))

    def join_cross(self, v):
        self.cross_list.append(v)
        self.cross_net = np.hstack([self.cross_net, random((self.cross_net.shape[0], v.shape[0]))])

    def exit_cross(self, v):
        for cross_v_item in self.cross_list:
            if cross_v_item == v:
                self.cross_list.remove(v)
                # TODO remove cross_net
                break

    def iterate(self):
        cross_v = np.negative(one)
        for cross_v_item in self.cross_list:
            cross_v = np.vstack([cross_v, cross_v_item])

        # 根据接入的神经元的最新输出组装出本次迭代的输入
        vector = np.vstack([self.output_v, cross_v])
        # 组装全连接矩阵和输入权重矩阵
        matrix = np.hstack([self.fc_net, self.cross_net])

        # 原始输出线性归一化
        output = matrix.dot(vector) - 1
        output = output / np.power(self.size, 0.5)
        output_v = np.minimum(output, np.ones((self.size, 1)))

        # 激活
        sign = output_v / np.abs(output_v)
        output_v = np.abs(output_v)
        output_v = np.maximum(output_v - self.offset_v, 0)
        output_v = np.minimum(output_v, 1)
        output_v = output_v * sign

        if output_v.shape != self.output_v.shape:
            print("[error] shape mismatch")

        # 输出
        self.output_v = output_v

        # TODO 更新权重
