import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from castle import CASTLE, Parameters

""" A collection of utility functions relating to CASTLE and its output analysis """

def plot_average_loss_1D(avg_loss, x_axis, x_label):
    plt.plot(x_axis, avg_loss, linewidth=2.0)
    plt.ylabel("Average Info Loss")
    plt.xlabel(x_label)
    plt.grid(True)
    plt.show()

def plot_average_loss_2D(avg_loss, x_axis, x_label, y_axis, y_label):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x_axis, y_axis, avg_loss, rstride=1, cstride=1, cmap='winter', edgecolor='none')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel('Average Infomation Loss')
    plt.show()

def handler(value: pd.Series):
    pass

def test_beta(file_name, beta_list):
    frame = pd.read_csv(file_name)
    headers = list(frame.columns.values)[1:-1]

    avg_loss_list = []

    for beta in beta_list:
        params = Parameters()

        params.k = 5
        params.delta = 10
        params.beta = beta
        params.mu = 10

        stream = CASTLE(handler, headers, "FareAmount", params)

        for (_, row) in frame.iterrows():
            stream.insert(row)

        clusters = stream.big_gamma

        cum_loss = 0
        for cluster in clusters:
            cum_loss += cluster.information_loss(stream.global_ranges)
        avg_loss = cum_loss / len(clusters)

        avg_loss_list.append(avg_loss)

    plot_average_loss_1D(avg_loss_list, beta_list, "Beta")


def test_k(file_name, k_list):
    frame = pd.read_csv(file_name)
    headers = list(frame.columns.values)[1:-1]

    avg_loss_list = []

    for k in k_list:
        params = Parameters()

        params.k = k
        params.delta = 10
        params.beta = 10
        params.mu = 10

        stream = CASTLE(handler, headers, "FareAmount", params)

        for (_, row) in frame.iterrows():
            stream.insert(row)

        clusters = stream.big_gamma

        cum_loss = 0
        for cluster in clusters:
            cum_loss += cluster.information_loss(stream.global_ranges)
        avg_loss = cum_loss / len(clusters)

        avg_loss_list.append(avg_loss)

    plot_average_loss_1D(avg_loss_list, k_list, "k")

def test_beta_mu(file_name, beta_list, mu_list):
    frame = pd.read_csv(file_name)
    headers = list(frame.columns.values)[1:-1]

    info_loss = []

    for mu in mu_list:
        print("mu: {}".format(mu))

        avg_loss_list = []

        for beta in beta_list:
            print("beta: {}".format(beta))
            params = Parameters()

            params.k = 10
            params.delta = 200
            params.beta = beta
            params.mu = mu
            params.l = 1
            params.dp = False

            stream = CASTLE(handler, headers, "FareAmount", params)

            for (_, row) in frame.iterrows():
                stream.insert(row)

            clusters = stream.big_gamma

            cum_loss = 0
            for cluster in clusters:
                cum_loss += cluster.information_loss(stream.global_ranges)
            avg_loss = cum_loss / len(clusters)
            avg_loss_list.append(avg_loss)

        info_loss.append(np.array(avg_loss_list))

    X, Y = np.meshgrid(beta_list, mu_list)
    plot_average_loss_2D(np.array(info_loss), X, "Beta", Y, "Mu")

