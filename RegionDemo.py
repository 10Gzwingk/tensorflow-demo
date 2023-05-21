import random

import AtomicRegion as ar
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
regionA.join_cross(regionB.cross_v, 'random')


fig, ax = plt.subplots()
ax.set_xlim(0, 200)
ax.set_ylim(0, 1)


def onclick(event):
    # 获取点击位置的x和y坐标
    x = int(event.xdata)
    y = event.ydata
    if x < 100:
        regionA.output_v[x - 1: x + 1, 0] = y
    else:
        regionB.output_v[x - 101: x - 99, 0] = y
    print(f"You clicked ({x}, {y})")


def animate(i):
    ax.clear()
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 1)
    # print(np.hstack((np.transpose(regionA.output_v)[0], np.transpose(regionB.output_v)[0])))
    ax.bar(range(200), np.hstack((np.transpose(regionA.output_v)[0], np.transpose(regionB.output_v)[0])))
    # regionA.output_v[0, 0] = random.random()
    # regionA.output_v[2, 0] = random.random()
    regionA.iterate()
    # regionA.output_v[0, 0] = random.random()
    # regionA.output_v[2, 0] = random.random()
    # regionA.iterate()
    regionB.iterate()


cid = fig.canvas.mpl_connect('button_press_event', onclick)

ani = FuncAnimation(fig, animate, frames=100, interval=20)
plt.show()


# for j in range(500):
#     regionA.output_v[0, 0] = random.random()
#     regionA.output_v[2, 0] = random.random()
#     regionA.iterate()
#     regionA.output_v[0, 0] = random.random()
#     regionA.output_v[2, 0] = random.random()
#     regionA.iterate()
#     regionB.iterate()
#     print(np.transpose(regionA.output_v))
# for i in range(500):
#     regionA.output_v[0, 0] = random.random()
#     regionA.output_v[2, 0] = random.random()
#     regionA.iterate()
#     regionA.output_v[0, 0] = random.random()
#     regionA.output_v[2, 0] = random.random()
#     regionA.iterate()
#     regionB.iterate()
#     print(np.transpose(regionA.output_v))
# for k in range(10000):
#     regionA.output_v[0, 0] = 1
#     regionA.output_v[2, 0] = 1
#     regionA.iterate()
#     regionA.output_v[0, 0] = 0
#     regionA.output_v[2, 0] = 0
#     regionA.iterate()
#     regionB.iterate()
#     print(np.transpose(regionB.output_v))
