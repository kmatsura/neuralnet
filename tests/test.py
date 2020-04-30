import numpy as np
import random
from pprint import pprint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# n = int(input())


def main():
    n = 7
    _graph = np.ones((n, n))
    graph = np.tril(_graph, k=-1)  # intilize edge
    _adjacent = np.ones((n, n))
    adjacent = np.tril(_adjacent, k=-1)  # adjacent matrix
    data_size = 3000
    data = make_sample_data(data_size)
    train_data = data[:int(len(data)*0.8)]
    x1 = data[0, :]
    x2 = data[1, :]
    y = data[2, :]
    plot3d(x1, x2, y)


def make_sample_data(number):
    D = np.zeros((3, number))
    for i in range(number):
        eps = random.random()
        x1 = (random.random() - 1) * 2
        x2 = random.random() * 3
        y = x1 * x2**2 + eps
        v = np.array([x1, x2, y])
        D[:, i] = v 
    return D


def plot3d(x1, x2, y):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    ax.plot(x1, x2, y, marker="o", markersize=1, linestyle='None')
    plt.savefig("sample_data.png")


if __name__ == "__main__":
    main()
