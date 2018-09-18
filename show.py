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
    axs[0].title('total energy')
    axs[1].hist(elec, bins='auto')
    axs[1].title('electric energy')
    axs[2].hist(ising, bins='auto')
    axs[2].title('ising energy')
    plt.show()
    fig.savefig('./images/hist.png')


def plot_loss_f(epochs, losses):
    plt.plot(epochs, losses)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('finer grid training loss')
    plt.show()
    plt.savefig("./images/loss_f.png")


def plot_loss_c(epochs, losses_tra, losses_val):
    plt.plot(epochs, losses_tra)
    plt.plot(epochs, losses_val)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('coarse grid training loss')
    plt.show()
    plt.savefig("./images/loss_c.png")


def plot_data(data, s_data, total, elec, ising):
    fig, axs = plt.subplots(len(data), 2)
    for i in range(len(data)):
        axs[i][0].imshow(data[i])
        title = str(round(total[i], 2)) + \
                str(round(elec[i], 2)) + str(round(ising[i], 2))
        axs[i][0].title(title)
        axs[i][0].axis("off")
        axs[i][1].imshow(s_data[i])
        axs[i][1].axis("off")
    plt.show()
    fig.savefig("./images/data.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', help='data saving directory')
    parser.add_argument('--f-losses', help='finer losses save file',
                        dest='finer_path')
    parser.add_argument('--c-losses', help='coarse losses save file',
                        dest='coarse_path')
    args = parser.parse_args()

    # plot energy hist and data image(only use test data)
    path = args.data + 'test-data.hdf5'
    f = h5py.File(path, 'r')
    energy_hist(f['energy'], f['elecenergy'], f['isenergy'])
    plot_data(f['ori_data'][:3], f['avg_data'][:3], f['energy'][:3],
              f['elecenergy'][:3], f['isenergy'][:3])

    # plot coarse model loss
    path = args.coarse_path
    f1 = h5py.File(path, 'r')
    plot_loss_c(f1['epochs'], f1['train'], f1['valid'])

    # plot finer model loss
    path = args.finer_path
    f2 = h5py.File(path, 'r')
    plot_loss_f(f2['epochs'], f2['losses'])