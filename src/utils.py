import numpy as np
import matplotlib
from matplotlib import pyplot as plt


def plot_results(LD, TS, ylab, eps, nr_trials, k_list):
    """
    Shows plot comparing results of TS and LD
    """
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)   # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig, ax = plt.subplots()
    ax.plot(k_list, LD, label="LD")
    ax.plot(k_list, TS, label="TS")
    ax.set(xlabel='k', ylabel=ylab, title='Comparison on ' + str(nr_trials) + ' trials, for epsilon of ' + str(eps))
    ax.grid()
    leg = ax.legend()
    plt.show()


def get_metrics(output, k, sorted_data):
    """
    Returns metrics P and S for a given output
    """
    return np.sum(np.array(output) <= k)/k, np.sum(sorted_data[output])/np.sum(sorted_data[:k])

