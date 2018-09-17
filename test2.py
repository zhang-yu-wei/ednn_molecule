# -*- coding: utf-8 -*-

from __future__ import print_function

import tensorflow as tf
import h5py
import argparse
from ednn import EDNN_helper
import os
import numpy as np
import matplotlib.pyplot as plt
# if this code is going to run on a server without GUI
plt.switch_backend('agg')


def NN_coarse(_in):
    tile_size = f_c + 2*c_c
    # default is 576
    _in = tf.reshape(_in, (-1, tile_size**2))
    nn = tf.contrib.layers.fully_connected(_in, 512, reuse=False, scope='ful1')
    nn = tf.contrib.layers.fully_connected(nn, 256, reuse=False, scope='ful2')
    nn = tf.contrib.layers.fully_connected(nn, 128, reuse=False, scope='ful3')
    nn = tf.contrib.layers.fully_connected(nn, 64, reuse=False, scope='ful4')
    nn = tf.contrib.layers.fully_connected(nn, 1, activation_fn=None, reuse=False,
                                           scope='ful5')
    return nn


def NN_finer(_in):
    tile_size = 3
    _in = tf.reshape(_in, (-1, tile_size**2))
    nn = tf.contrib.layers.fully_connected(_in, 9, reuse=False, scope='ful1')
    nn = tf.contrib.layers.fully_connected(nn, 5, reuse=False, scope='ful2')
    nn = tf.contrib.layers.fully_connected(nn, 1, activation_fn=None, reuse=False,
                                           scope='ful3')
    return nn


def build_model(L, f, c, test_data, save_dir, type):
    if type == 'coarse':
        coarse = tf.Graph()
        with coarse.as_default():
            x = tf.placeholder(tf.float32, (None, L, L), name='input_image')
            helper = EDNN_helper(L=L, f=f, c=c)
            tiles = tf.map_fn(helper.ednn_split, x, back_prop=False)
            tiles = tf.transpose(tiles, perm=[1, 0, 2, 3, 4])
            with tf.variable_scope('train') as scope:
                # Then the EDNN-specific code:
                output = tf.map_fn(NN_coarse, tiles, back_prop=True)
                output = tf.transpose(output, perm=[1, 0, 2])
                predicted = tf.reduce_sum(output, axis=1)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.InteractiveSession(config=config)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            model_path = os.path.join(save_dir, 'model')
            saver.restore(sess, model_path)
            pred = sess.run(predicted, feed_dict={x: test_data})
            sess.close()
        return pred

    if type == 'finer':
        finer = tf.Graph()
        with finer.as_default():
            x = tf.placeholder(tf.float32, (None, 3, 3), name='input_image')
            helper = EDNN_helper(L=L, f=f, c=c)
            tiles = tf.map_fn(helper.ednn_split, x, back_prop=False)
            tiles = tf.transpose(tiles, perm=[1, 0, 2, 3, 4])

            output = tf.map_fn(NN_finer, tiles, back_prop=True)
            output = tf.transpose(output, perm=[1, 0, 2])
            predicted = tf.reduce_sum(output, axis=1)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.InteractiveSession(config=config)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            model_path = os.path.join(save_dir, 'model')
            saver.restore(sess, model_path)
            pred = sess.run(predicted, feed_dict={x: test_data})
            sess.close()
        return pred

    else:
        print("model type is not right")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--c-save', help='coarse model saved directory',
                        dest='save_c')
    parser.add_argument('--f-save', help='finer model saved directory',
                        dest='save_f')
    parser.add_argument('data', help='data directory')

    parser.add_argument('-L', help='data size', type=int,
                        default=32, dest='L')
    parser.add_argument('-f', help='focus size of coarse model',
                        type=int, default=8, dest='f_c')
    parser.add_argument('-c', help='context size of coarse model',
                        type=int, default=8, dest='c_c')

    args = parser.parse_args()

    f_c = args.f_c
    c_c = args.c_c

    path = args.data + '/test-data' + '/data.hdf5'
    f = h5py.File(path, 'r')

    # load data
    data_f = f['ori_data']
    data_c = f['avg_data']
    total_energy = np.reshape(f['energy'], [-1, 1])
    elec_energy = np.reshape(f['elecenergy'], [-1, 1])
    is_energy = np.reshape(f['isenergy'], [-1, 1])

    elec_pred = build_model(args.L, f_c, c_c, data_c, args.save_c, 'coarse')
    is_pred = build_model(3, 1, 1, data_f, args.save_f, 'finer')
    total_pred = [a + b for a, b in elec_pred, is_pred]

    fig, axs = plt.subplots(3, 1, figsize=(10, 20))
    axs[0].plot(total_pred, total_energy, '.')
    axs[0].set_title("total energy")
    axs[0].set_xlabel("prediction")
    axs[0].set_ylabel("real energy")
    axs[1].plot(elec_pred, elec_energy, '.')
    axs[1].set_title("electric energy")
    axs[1].set_xlabel("prediction")
    axs[1].set_ylabel("real energy")
    axs[2].plot(is_pred, is_energy, '.')
    axs[2].set_title("ising energy")
    axs[2].set_xlabel("prediction")
    axs[2].set_ylabel("real energy")
    plt.savefig('prediction.png')