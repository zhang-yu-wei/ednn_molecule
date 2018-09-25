# -*- coding: utf-8 -*-

from __future__ import print_function

import matplotlib.pyplot as plt
import h5py
import argparse
# if this code is going to run on a server without GUI
plt.switch_backend('agg')

"""
This code is used to visualize the results.
What is being visualized:
1. what the data look like
2. energy histogram
3. losses with epochs
"""


def energy_hist(total, elec, ising):
    fig, axs = plt.subplots(3, 1, figsize=(5, 15))

    axs[0].hist(total, bins='auto')
    axs[0].set_title('total energy')
    axs[1].hist(elec, bins='auto')
    axs[1].set_title('electric energy')
    axs[2].hist(ising, bins='auto')
    axs[2].set_title('ising energy')
    plt.show()
    fig.savefig('./images/hist.png')
    plt.close()


def plot_loss_f(epochs, losses):
    plt.plot(epochs, losses)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('finer grid training loss')
    plt.show()
    plt.savefig("./images/loss_f.png")
    plt.close()


def plot_loss_c(epochs, losses_tra, losses_val):
    plt.plot(epochs, losses_tra)
    plt.plot(epochs, losses_val)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('coarse grid training loss')
    plt.show()
    plt.savefig("./images/loss_c.png")
    plt.close()


def plot_data(data, s_data, total, elec, ising):
    fig, axs = plt.subplots(len(data), 2)
    for i in range(len(data)):
        axs[i][0].imshow(data[i])
        title = str(round(total[i], 2)) + \
                str(round(elec[i], 2)) + str(round(ising[i], 2))
        axs[i][0].set_title(title)
        axs[i][0].axis("off")
        axs[i][1].imshow(s_data[i])
        axs[i][1].axis("off")
    plt.show()
    fig.savefig("./images/data.png")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', help='data saving directory')
    args = parser.parse_args()

    # plot energy hist and data image(only use test data)
    path = args.data + '/test-data/energy.hdf5'
    f1 = h5py.File(path, 'r')
    energy_hist(f1['total'], f1['elec'], f1['ising'])
    path = args.data + '/test-data/ori.hdf5'
    f2 = h5py.File(path, 'r')
    path = args.data + '/test-data/avg.hdf5'
    f3 = h5py.File(path, 'r')
    plot_data(f2['ori_data'][:3], f3['avg_data'][:3], f1['total'][:3],
              f1['elec'][:3], f1['ising'][:3])

    # plot coarse model loss
    f4 = h5py.File('coarse_losses.hdf5', 'r')
    plot_loss_c(f4['epochs'], f4['train'], f4['valid'])

    # plot finer model loss
    f5 = h5py.File('finer_losses.hdf5', 'r')
    plot_loss_f(f5['epochs'], f5['losses'])
