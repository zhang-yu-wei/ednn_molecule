import tensorflow as tf
from ednn_helper import EDNN_helper
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def NN_coarse(_in):
    tile_size = f_c + 2*c_c
    _in = tf.reshape(_in, (-1, tile_size**2))
    nn = tf.contrib.layers.fully_connected(_in, 512, scope='ful1')
    nn = tf.contrib.layers.fully_connected(nn, 256, scope='ful2')
    nn = tf.contrib.layers.fully_connected(nn, 128, scope='ful3')
    nn = tf.contrib.layers.fully_connected(nn, 64, scope='ful4')
    nn = tf.contrib.layers.fully_connected(nn, 1, activation_fn=None, scope='ful5')
    return nn


def build_model(L, f, c, test_data, save_dir):
  b = tf.Graph()
  with b.as_default():
    x = tf.placeholder(tf.float32, (None, L, L), name='input_image')
    helper = EDNN_helper(L=L, f=f, c=c)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    tiles = tf.map_fn(helper.ednn_split, x, back_prop=False)
    tiles = tf.transpose(tiles, perm=[1, 0, 2, 3, 4])
    with tf.variable_scope('train') as scope:
        # Then the EDNN-specific code:
        output = tf.map_fn(NN_coarse, tiles, back_prop=True)
        output = tf.transpose(output, perm=[1, 0, 2])
        predicted = tf.reduce_sum(output, axis=1)
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    sess.run(init)
    saver = tf.train.Saver()
    model_path = os.path.join(save_dir, 'model')
    saver.restore(sess, model_path)
    pred = sess.run(predicted, feed_dict={x:test_data})
    sess.close()
  return pred


if __name__ == '__main__':
  L = 32
  f_c = 8
  c_c = 8

  f1 = h5py.File('train-data2/data.hdf5', 'r')
  f2 = h5py.File('valid-data2/data.hdf5', 'r')
  f3 = h5py.File('test-data2/data.hdf5', 'r')

  train_data = f1['data'][100:200]
  train_labels = np.reshape(f1['elecenergy'], [-1, 1])[100:200]
  test_data = f3['data']
  test_labels = np.reshape(f3['elecenergy'], [-1, 1])

  train_pred = build_model(L, f_c, c_c, train_data, 'coarse_model')
  test_pred = build_model(L, f_c, c_c, test_data, 'coarse_model')

  fig, axs = plt.subplots(2, 1, figsize=(10, 20))
  axs[0].plot(train_pred, train_labels, '.')
  axs[0].set_title("train")
  axs[0].set_xlabel("prediction")
  axs[0].set_ylabel("real energy")
  axs[1].plot(test_pred, test_labels, '.')
  axs[1].set_title("test")
  axs[1].set_xlabel("prediction")
  axs[1].set_ylabel("real energy")
  plt.show()
  plt.savefig('test2.png')

