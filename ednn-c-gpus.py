import tensorflow as tf
from ednn_helper import EDNN_helper
import h5py
import os
import numpy as np


def NN_coarse(_in):
    tile_size = f_c + 2*c_c
    _in = tf.reshape(_in, (-1, tile_size**2))
    nn = tf.contrib.layers.fully_connected(_in, 512, reuse=False, scope='ful1')
    nn = tf.contrib.layers.fully_connected(nn, 256, reuse=False, scope='ful2')
    nn = tf.contrib.layers.fully_connected(nn, 128, reuse=False, scope='ful3')
    nn = tf.contrib.layers.fully_connected(nn, 64, reuse=False, scope='ful4')
    nn = tf.contrib.layers.fully_connected(nn, 1, activation_fn=None, reuse=False, scope='ful5')
    return nn


def average_tower_grads( tower_grads):
    if(len(tower_grads) == 1):
      return tower_grads[0]
    avgGrad_var_s = []
    for grad_var_s in zip(*tower_grads):
      grads = []
      v = None
      for g, v_ in grad_var_s:
        g = tf.expand_dims(g, 0)
        grads.append(g)
        v = v_
      all_g = tf.concat(grads, 0)
      avg_g = tf.reduce_mean(all_g, 0, keep_dims=False)
      avgGrad_var_s.append((avg_g, v))
    return avgGrad_var_s


def build_model(L, f, c, train_data, train_labels, valid_data,
                valid_labels, save_dir, summary_dir):
    a = tf.Graph()
    with a.as_default():
      # data comes in a [ batch * L * L ] tensor, and labels a [ batch * 1] tensor
      with tf.device("/cpu:0"):
        x = tf.placeholder(tf.float32, (2, None, L, L), name='input_image')
        y = tf.placeholder(tf.float32, (2, None, 1))
        helper = EDNN_helper(L=L, f=f, c=c)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        towerGrads = []
        towerloss = []
      with tf.variable_scope('train') as scope:
        for i in range(2):
            with tf.device("/gpu:%d" % i):
                with tf.name_scope('tower_%d' % i) as scope:
                    # Then the EDNN-specific code:
                    tiles = tf.map_fn(helper.ednn_split, x[i], back_prop=False)
                    tiles = tf.transpose(tiles, perm=[1, 0, 2, 3, 4])
                    output = tf.map_fn(NN_coarse, tiles, back_prop=True)
                    output = tf.transpose(output, perm=[1, 0, 2])
                    predicted = tf.reduce_sum(output, axis=1)
                    # define the loss function
                    vars = tf.trainable_variables()
                    lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars
                                       if 'bias' not in v.name]) * 0.001
                    loss_ = tf.reduce_mean(tf.square(y[i] - predicted)) + lossL2
                    towerloss.append(loss_)
                    towerGrads.append(optimizer.compute_gradients(loss_))
                    tf.get_variable_scope().reuse_variables()
      avg_Grads = average_tower_grads(towerGrads)
      train_step = optimizer.apply_gradients(avg_Grads)
      loss = tf.reduce_sum(towerloss) / 2
      tf.summary.scalar('loss', loss)
      init = tf.global_variables_initializer()
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      sess = tf.InteractiveSession(config=config)
      merged = tf.summary.merge_all()
      train_writer = tf.summary.FileWriter(summary_dir + '/train', sess.graph)
      valid_writer = tf.summary.FileWriter(summary_dir + '/valid')
      sess.run(init)
      saver = tf.train.Saver()
      a.finalize()
    model_path = os.path.join(save_dir, 'model')
    BATCH_SIZE = 5000
    EPOCHS = 5000
    for epoch in range(EPOCHS):
        for batch in range(int(train_data.shape[0] / BATCH_SIZE / 2)):
            _, loss_tra, summary1 = sess.run([train_step, loss, merged],
                       feed_dict={
                            x: [train_data[2*batch*BATCH_SIZE:(2*batch+1)*BATCH_SIZE], train_data[(2*batch+1)*BATCH_SIZE:2*(batch+1)*BATCH_SIZE]], 
                            y: [train_labels[2*batch*BATCH_SIZE:(2*batch+1)*BATCH_SIZE], train_labels[(2*batch+1)*BATCH_SIZE:2*(batch+1)*BATCH_SIZE]]
                        }
                      )

            if batch % 1000 == 0:
                loss_val, summary2 = sess.run([loss, merged], 
                                    feed_dict={
                                        x: [valid_data[:int(len(valid_data)/2)], valid_data[int(len(valid_data)/2):]], 
                                        y: [valid_labels[:int(len(valid_data)/2)], valid_labels[int(len(valid_data)/2):]]
                                    }
                                    )
                train_writer.add_summary(summary1, epoch)
                valid_writer.add_summary(summary2, epoch)
                print "epoch: " + str(epoch) + ' | training loss: ' + str(loss_tra) \
                      + ' | validation loss: ' + str(loss_val) + " (model saved)"
                saver.save(sess, model_path)

if __name__ == '__main__':
    L = 32
    f_c = 8
    c_c = 8

    f1 = h5py.File('train-data3/data.hdf5', 'r')
    f2 = h5py.File('valid-data3/data.hdf5', 'r')
    f3 = h5py.File('test-data3/data.hdf5', 'r')

    train_data = f1['data']
    train_labels = np.reshape(f1['elecenergy'], [-1, 1])
    valid_data = f2['data']
    valid_labels = np.reshape(f2['elecenergy'], [-1, 1])

    build_model(L, f_c, c_c, train_data, train_labels, valid_data,
                    valid_labels, 'coarse_model', 'summary')
