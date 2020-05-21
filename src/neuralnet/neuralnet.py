import numpy as np
import random
from pprint import pprint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import gc
import math
import pickle
import os

def main():
    make_matrix([2, 4, 3, 1])

def make_matrix(node_list):
    '''
    Make adjacent matrix of neural network from node list.
    The head of the list must be the input vector dimensions and tail must be
    the output.
    '''
    node_que = deque(node_list)
    total_nodes = sum(node_que)
    adjacent = np.zeros((total_nodes, total_nodes))  # make adjacent matrix

    def insert_node(adjacent, node_que, row, string, pointer):
        if node_que:
            string = row
            row = node_que.popleft()
            adjacent[pointer[0]:pointer[0]+row, pointer[1]:pointer[1]+string]\
                = np.ones((row, string))
            pointer = (pointer[0]+row, pointer[1]+string)
            return insert_node(adjacent, node_que, row, string, pointer)
        else:
            return adjacent
    row = node_que.popleft()
    adjacent = insert_node(adjacent, node_que, row, 0, (row, 0))
    print(adjacent)


def sample_learning():
    outputdir = "./result/sample_learning2/"
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    nodes_number = 7  # the number of nodes
    hidden_layer = 2
    _graph = np.random.randn(nodes_number, nodes_number) * 2 - 1
    _adjacent = np.ones((nodes_number, nodes_number))
    adjacent = np.tril(_adjacent, k=-1)  # adjacent matrix
    #  make layers
    adjacent[1, 0] = 0
    adjacent[3, 2] = 0
    adjacent[4:, 0:2] = np.zeros((3, 2))
    adjacent[5, 4] = 0
    adjacent[6, 2:4] = np.zeros(2)
    graph_init = _graph * adjacent  # intilize edge
    data_size = 1000
    data = make_sample_data(data_size)
    train_rate = 0.8
    train_data = data[:, :int(data.shape[1]*train_rate)]
    test_data = data[:, int(data.shape[1]*train_rate):]
    X_train = train_data[0:2, :]
    y_train = train_data[2, :]
    X_test = test_data[0:2, :]
    y_test = test_data[2, :]
    plot3d(data[0, :], data[1, :], data[2, :], outputdir + "sample.png")
    epoch = 100
    activate = leaky_relu
    activate_diff = leaky_relu_diff
    run(epoch, graph_init, nodes_number, hidden_layer, X_train,
        y_train, X_test, y_test, activate, activate_diff, outputdir)


def graph_init():
    return graph_init


def run(epoch, graph_init, nodes_number, hidden_layer, X_train,
        y_train, X_test, y_test, activate, activate_diff, outputdir):
    train_loss_hist = []
    test_loss_hist = []
    for i in range(epoch):
        train_loss, test_loss = run_iteration(i, graph_init, nodes_number, hidden_layer, X_train,
                                              y_train, X_test, y_test, activate, activate_diff, outputdir)
        train_loss_hist.append(train_loss)
        test_loss_hist.append(test_loss)
    with open('train_loss_hist.txt', 'wb') as f:
        pickle.dump(train_loss_hist, f)
    with open('test_loss_hist.txt', 'wb') as f:
        pickle.dump(test_loss_hist, f)


def run_iteration(epoch, graph_init, nodes_number, hidden_layer, X_train,
                  y_train, X_test, y_test, activate, activate_diff, outputdir):
    graph, train_loss = learing_parameters(graph_init, nodes_number,
                                           hidden_layer, X_train, y_train,
                                           activate, activate_diff)
    test_loss, y_test_pred = calc_test_loss(
        graph, X_test, y_test, nodes_number, hidden_layer, activate)
    print("epoch:{0}, test loss:{1}, train loss:{2}".format(
        epoch+1, test_loss, train_loss))
    plot3d(X_test[0, :], X_test[1, :], y_test_pred,
           outputdir + "prediction_epoch{}.png".format(epoch+1))
    return train_loss, test_loss


def calc_test_loss(graph, X_test, y_test, nodes_number, hidden_layer, activate):
    X__test_padding = make_padding(X_test, nodes_number, hidden_layer)
    activate = np.vectorize(activate)
    x_input = np.zeros((7, 1))  # initialize input vector
    test = len(y_test)
    y_test_pred = np.zeros(test)
    x__test_output_hist = np.zeros(X__test_padding.shape)
    x_test_tmp_hist = np.zeros(X__test_padding.shape)
    for j in range(test + hidden_layer):
        x_input += X__test_padding[:, j].reshape(7, 1)
        x_tmp = np.dot(graph, x_input)
        x_test_tmp_hist[:, j] = x_tmp.reshape(7)
        x_output = activate(x_tmp)
        x__test_output_hist[:, j] = x_output.reshape(7)
        x_input = x_output
        if j >= hidden_layer:
            y_test_pred[j-hidden_layer] = x_tmp[6, 0]
    test_loss = calc_loss(squared_error, y_test_pred, y_test)
    return test_loss, y_test_pred


def learing_parameters(graph, nodes_number, hidden_layer, X, y, activate,
                       activate_diff, alpha=0.01, sample=300):
    if sample > X.shape[1]:
        sample = X.shape[1]
    use_list = random.sample(list(range(X.shape[1])), sample)
    X_use = np.zeros((X.shape[0], sample))
    y_use = np.zeros(sample)
    y_use_pred = np.zeros(sample)
    for i, num in enumerate(use_list):
        X_use[:, i] = X[:, num]
        y_use[i] = y[num]
    X_padding = make_padding(X_use, nodes_number, hidden_layer)
    activate = np.vectorize(activate)
    x_output_hist = np.zeros(X_padding.shape)
    x_tmp_hist = np.zeros(X_padding.shape)
    x_input = np.zeros((graph.shape[0], 1))  # initialize input vector
    for i in range(sample + hidden_layer):
        pprint(graph)
        x_input += X_padding[:, i].reshape(7, 1)
        x_tmp = np.dot(graph, x_input)
        x_tmp_hist[:, i] = x_tmp.reshape(7)
        x_output = activate(x_tmp)
        x_output_hist[:, i] = x_output.reshape(7)
        x_input = x_output
        if i >= hidden_layer:
            y_use_pred[i-hidden_layer] = x_tmp[6, 0]
            delta = y_use_pred[i-hidden_layer] - y_use[i-hidden_layer]
            #  update parameters
            same = alpha * 2 * delta
            graph[2, 0] -= same * (graph[6, 4]
                                   * activate_diff(x_tmp_hist[4, i-1])
                                   * graph[4, 2]
                                   * activate_diff(x_tmp_hist[2, i-2])
                                   * X_padding[0, i-hidden_layer]
                                   + graph[6, 5]
                                   * activate_diff(x_tmp_hist[5, i-1])
                                   * graph[5, 2]
                                   * activate_diff(x_tmp_hist[2, i-2])
                                   * X_padding[0, i-hidden_layer])
            graph[2, 1] -= same * (graph[6, 4]
                                   * activate_diff(x_tmp_hist[4, i-1])
                                   * graph[4, 2]
                                   * activate_diff(x_tmp_hist[2, i-2])
                                   * X_padding[1, i-hidden_layer]
                                   + graph[6, 5]
                                   * activate_diff(x_tmp_hist[5, i-1])
                                   * graph[5, 2]
                                   * activate_diff(x_tmp_hist[2, i-2])
                                   * X_padding[1, i-hidden_layer])
            graph[3, 0] -= same * (graph[6, 4]
                                   * activate_diff(x_tmp_hist[4, i-1])
                                   * graph[4, 3]
                                   * activate_diff(x_tmp_hist[3, i-2])
                                   * X_padding[0, i-hidden_layer]
                                   + graph[6, 5]
                                   * activate_diff(x_tmp_hist[5, i-1])
                                   * graph[5, 3]
                                   * activate_diff(x_tmp_hist[3, i-2])
                                   * X_padding[0, i-hidden_layer])
            graph[3, 1] -= same * (graph[6, 4]
                                   * activate_diff(x_tmp_hist[4, i-1])
                                   * graph[4, 3]
                                   * activate_diff(x_tmp_hist[3, i-2])
                                   * X_padding[1, i-hidden_layer]
                                   + graph[6, 5]
                                   * activate_diff(x_tmp_hist[5, i-1])
                                   * graph[5, 3]
                                   * activate_diff(x_tmp_hist[3, i-2])
                                   * X_padding[1, i-hidden_layer])

            graph[4, 2] -= same * graph[6, 4] * \
                activate_diff(
                    x_tmp_hist[4, i-1]) * x_output_hist[2, i-2]
            graph[4, 3] -= same * graph[6, 4] * \
                activate_diff(
                    x_tmp_hist[4, i-1]) * x_output_hist[3, i-2]
            graph[5, 2] -= same * graph[6, 5] * \
                activate_diff(
                    x_tmp_hist[5, i-1]) * x_output_hist[2, i-2]
            graph[5, 3] -= same * graph[6, 5] * \
                activate_diff(
                    x_tmp_hist[5, i-1]) * x_output_hist[3, i-2]
            graph[6, 4] -= same * x_output_hist[4, i-1]
            graph[6, 5] -= same * x_output_hist[5, i-1]
    loss = calc_loss(squared_error, y_use_pred, y_use)
    del use_list
    del X_use
    del y_use
    del y_use_pred
    del x_output_hist
    del x_tmp_hist
    del X_padding
    gc.collect()
    return graph, loss


def make_padding(X, nodes_number, hidden_layer):
    padding1 = np.zeros((nodes_number - X.shape[0], X.shape[1]))
    tmp_vector = np.concatenate([X, padding1], axis=0)
    padding2 = np.zeros((nodes_number, hidden_layer))
    input_vector = np.concatenate([tmp_vector, padding2], axis=1)
    return input_vector


def calc_loss(loss, y_pred, y):
    ans = 0
    for i in range(len(y)):
        ans += squared_error(y_pred[i], y[i])
    ans /= float(len(y))
    return ans


def squared_error(y_pred, y):
    return (y_pred - y) ** 2


def sigmoid(x):
    return 1 / (1. + np.exp(-x))


def sigmoid_diff(x):
    return 1 / (np.exp(x/2) + np.exp(-x/2))**2


def leaky_relu(x, a=0.01):
    if x > 0.:
        return x
    else:
        return a * x


def leaky_relu_diff(x, a=0.01):
    if x > 0:
        return 1
    else:
        return a


def relu(x):
    if x > 0.:
        return x
    else:
        return 0.


def relu_diff(x):
    if x > 0.:
        return 1
    else:
        return 0.


def make_sample_data(number):
    D = np.zeros((3, number))
    for i in range(number):
        eps = random.random()
        x1 = (random.random() - 1) * 5
        x2 = (random.random() - 1) * 5
        y = 5 * math.sin(x2) + x1 + eps
        D[:, i] = np.array([x1, x2, y])
    return D


def plot3d(x1, x2, y, filename):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    ax.plot(x1, x2, y, marker="o", markersize=1, linestyle='None')
    plt.savefig(filename)
    plt.close()
    del fig
    del ax
    gc.collect()


if __name__ == "__main__":
    main()
