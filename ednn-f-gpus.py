# -*- coding: utf-8 -*-

from __future__ import print_function

import tensorflow as tf
import h5py
import argparse
import numpy as np
from ednn_helper import EDNN_helper
import os
import h5py
from progress import progress_timer


"""
This file is used to train finer model
"""


def NN_finer(_in):
    tile_size = 6
    inputs = []
    for tile in _in:
        if tf.count_nonzero(tile) ==0:
           continue
        else:
           inputs.append(tile)
    inputs = tf.reshape(inputs, (-1, tile_size**2))
    #_in = tf.reshape(_in, (-1, tile_size**2))
    nn = tf.contrib.layers.fully_connected(_in, 32, reuse=False, scope='ful1')
    nn = tf.contrib.layers.fully_connected(inputs, 32, reuse=False, scope='ful1')
    nn = tf.contrib.layers.fully_connected(nn, 64, reuse=False, scope='ful2')
    nn = tf.contrib.layers.fully_connected(nn, 1, activation_fn=None, reuse=False,
                                           scope='ful3')
    return nn


def H(arr):
   #shift one to the right elements
   x = np.roll(arr,1,axis=1)
   #shift elements one down
   y = np.roll(arr,1,axis=0)
   #multiply original with transformations and sum each arr
   x = np.sum(np.multiply(arr,x))
   y = np.sum(np.multiply(arr,y))
   return -float(x+y)


def build_model(train_data, train_labels, valid_data, valid_labels, save_dir, BATCH_SIZE=1000,
                EPOCHS=100):
    with tf.get_default_graph().as_default():
        x = tf.placeholder(tf.float32, (None, 256, 256), name='input_image')
        y = tf.placeholder(tf.float32, (None, 1))

        helper = EDNN_helper(L=256, f=4, c=1)
        with tf.device("/cpu:0"):
            #Then the EDNN-specific code:
            tiles = tf.map_fn(helper.ednn_split, x, back_prop=False)
            tiles = tf.transpose(tiles, perm=[1,0,2,3,4])
            output = tf.map_fn(NN_finer, tiles, back_prop=True)
            output = tf.transpose(output, perm=[1,0,2])
            predicted = tf.reduce_sum(output, axis=1)

            #define the loss function
            loss = tf.reduce_mean(tf.square(y-predicted))

            #create an optimizer, a training op, and an init op
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_step = optimizer.minimize(loss)

        # define session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.InteractiveSession(config=config)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        #constants
        model_path = os.path.join(save_dir, 'model')
        losses_tra = []
        losses_val = []
        epochs = []

        for epoch in range(EPOCHS):
            for batch in range(int(np.shape(train_data)[0] / BATCH_SIZE)):
                _, loss_tra = sess.run([train_step, loss],
                       feed_dict={
                            x: train_data[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE],
                            y: train_labels[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]
                        }
                      )
                loss_val = sess.run(loss, 
                       feed_dict={
                            x: valid_data, 
                            y: valid_labels
                        }
                      )
                print("epoch: " + str(epoch) + ' | training loss: '
                          + str(loss_tra) + ' | validation loss: ' + str(loss_val))
            saver.save(sess, save_path=model_path)
            print('model saved')
            losses_tra.append(loss_tra)
            losses_val.append(loss_val)
            epochs.append(epoch)
    return epochs, losses_tra, losses_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('save', help='model save directory')
    parser.add_argument('data', help='data save directory')
    parser.add_argument('-b', help='batch size(default: 1000)', dest='BATCH_SIZE',
                        type=int, default=1000)
    parser.add_argument('-e', help='epochs(default: 1000)', dest='EPOCHS',
                        type=int, default=1000)

    args = parser.parse_args()

    path = args.data + '/train-data/ori.hdf5'
    f1 = h5py.File(path)
    train_data = f1['ori_data'][:10000]

    path = args.data + '/valid-data/ori.hdf5'
    f2 = h5py.File(path)
    valid_data = f2['ori_data']

    path = args.data + '/train-data/energy.hdf5'
    f3 = h5py.File(path)
    train_labels = np.reshape(f3['ising'][:10000], [-1, 1])

    path = args.data + '/valid-data/energy.hdf5'
    f4 = h5py.File(path)
    valid_labels = np.reshape(f4['ising'][:10000], [-1, 1])

    epochs, losses_tra, losses_val = build_model(train_data, train_labels, valid_data, valid_labels, args.save, args.BATCH_SIZE, args.EPOCHS)

    loss_f = h5py.File('finer_losses.hdf5', 'w')
    loss_f['epochs'] = epochs
    loss_f['losses_tra'] = losses_tra
    loss_f['losses_val'] = losses_val




