# -*- coding: utf-8 -*-

from __future__ import print_function

import tensorflow as tf
import h5py
import argparse
import numpy as np
from ednn_helper import EDNN_helper
import os


def NN_finer(_in):
    tile_size = 3
    _in = tf.reshape(_in, (-1, tile_size**2))
    nn = tf.contrib.layers.fully_connected(_in, 9, reuse=False, scope='ful1')
    nn = tf.contrib.layers.fully_connected(nn, 5, reuse=False, scope='ful2')
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


def get_data(num_exm):
    data = []
    label = []
    for _ in range(num_exm):
        grid = np.random.uniform(size=[3, 3])
        grid = np.round(grid) * 2 - 1
        data.append(grid)
        label.append(H(grid))
    return data, label


def build_model(train_data, train_labels, save_dir, report=10, BATCH_SIZE=1000,
                EPOCHS=100):
    with tf.get_default_graph.as_default():
        x = tf.placeholder(tf.float32, (None, 3, 3), name='input_image')
        y = tf.placeholder(tf.float32, (None, 1))

        helper = EDNN_helper(L=3, f=1, c=1)

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
        losses = []
        epochs = []

        for epoch in range(EPOCHS):
            for batch in range(train_data.shape[0] / BATCH_SIZE):
                _, loss_val = sess.run([train_step, loss],
                       feed_dict={
                            x: train_data[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE],
                            y: train_labels[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]
                        }
                      )

                if epoch % report ==0:
                    saver.save(sess, save_path=model_path)
                    print("epoch: " + str(epoch) + ' | training loss: '
                          + str(loss_val) + " (model saved)")
                    losses.append(loss_val)
                    epochs.append(epoch)
    return epochs, losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('save', help='save directory')

    parser.add_argument('-b', help='batch size', dest='BATCH_SIZE',
                        type=int, default=1000)
    parser.add_argument('-e', help='epochs', dest='EPOCHS',
                        type=int, default=100)
    parser.add_argument('-r', help='report interval', dest='report',
                        type=int, default=10)
    args = parser.parse_args()

    data, labels = get_data(10000)

    epochs, losses = build_model(data, labels, args.save)

    loss_f = h5py.File('finer_losses.hdf5', 'w')
    loss_f['epochs'] = epochs
    loss_f['losses'] = losses




