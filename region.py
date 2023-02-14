import math

import numpy as np

one = np.ones([1, 1])


def random(size):
    return (np.random.random(size) - 0.5) * 2


class Region:

    def __init__(self, size):
        self.cross_list = []
        self.size = size
        self.fc_net = random((size, size))  # 当前 region 的全连接矩阵
        self.step = random((size, size))    # 神经元记忆步长
        self.base = random((size, size))    # 神经元记忆偏置，输入信号大于 base 的才进行强化，小于 base 的则弱化
        self.output_v = random((size, 1))   # 当前 region 的神经元输出向量，维度不可变
        self.offset_v = random((size, 1))
        self.offset_base = random((size, 1))
        self.offset_step = random((size, 1))

    def join_cross(self, v):
        self.cross_list.append(v)
        self.fc_net = np.hstack([self.fc_net, random((self.fc_net.shape[0], v.shape[0]))])
        self.step = np.hstack([self.step, random((self.step.shape[0], v.shape[0]))])
        self.base = np.hstack([self.base, random((self.base.shape[0], v.shape[0]))])

    def exit_cross(self, v):
        for cross_v_item in self.cross_list:
            if cross_v_item == v:
                self.cross_list.remove(v)
                # TODO remove cross_net
                break

    def iterate(self):
        # 组装其他 region 的输入向量
        # 根据接入的神经元的最新输出组装出本次迭代的输入
        vector = self.output_v
        for cross_v_item in self.cross_list:
            vector = np.vstack([vector, cross_v_item])

        # 原始输出，由二次非归一，转为线性归一化
        output = self.fc_net.dot(vector) - 1
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

        # 基于本次的输入更新权重
        self.fc_net = np.arctan(
            (np.ones((self.size, 1)) * np.transpose(vector) - self.base) * self.step + np.tan(self.fc_net * math.pi / 2)
        ) / (math.pi / 2)
        # 根据输出更新偏移
        # self.offset_v = self.update(self.offset_v, self.output_v)
        self.fc_net = np.arctan(
            (self.output_v - self.base) * self.offset_step + np.tan(self.offset_v * math.pi / 2)
        ) / (math.pi / 2)

    def update(self, x, v):
        return np.arctan((np.ones((self.size, 1)) * np.transpose(v) - self.base) * self.step + np.tan(x * math.pi / 2)) / (math.pi / 2)
