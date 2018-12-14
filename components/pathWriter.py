import tensorlayer as tl
import tensorflow as tf
import config

class PathWriter():
    def __init__(self, batchSize):
        self.batchSize = batchSize
        self.lstm_cell = tf.nn.rnn_cell.LSTMCell(128, state_is_tuple=True)
        self.state = self.lstm_cell.zero_state(batchSize, tf.float32)

    def __call__(self, glimpse):
        with tf.variable_scope('pathWriter', reuse = tf.AUTO_REUSE):

            glimpse = tl.layers.InputLayer(glimpse)
            glimpse = tl.layers.DenseLayer(glimpse, n_units = 256, name ='glimpse_fc_1')
            glimpse = tl.layers.BatchNormLayer(glimpse, name='glimpse_bn_1')
            glimpse = tf.nn.relu(glimpse.outputs, name='glimpse_relu_1')

            self.output, self.state = self.lstm_cell(glimpse, self.state)

            net = tl.layers.InputLayer(self.state)
            net = tl.layers.DenseLayer(net, n_units = 128, name='fc_out1')
            net = tl.layers.BatchNormLayer(net, name='bn_1')
            net = tf.nn.relu(net.outputs, name='relu_1')

            net1 = tl.layers.DenseLayer(net, n_units = 2, name='fc_out1')
            net1 = tl.layers.BatchNormLayer(net1, name='bn_1')
            net1 = tf.nn.relu(net1.outputs, name='relu_1')

            net2 = tl.layers.DenseLayer(net, n_units = 2, name='fc_out2')
            net2 = tl.layers.BatchNormLayer(net2, name='bn_2')
            net2 = tf.nn.relu(net2.outputs, name='relu_2')

            net1Mean = tf.stop_gradient(tf.clip_by_value(net1, -1.0, 1.0))
            net1Location = net1Mean + tf.random_normal((self.batchSize, config.loc_dim), stddev=config.loc_std)
            net1Location = tf.stop_gradient(net1Location)

            net2Mean = tf.stop_gradient(tf.clip_by_value(net2, -1.0, 1.0))
            net2Location = net2Mean + tf.random_normal((self.batchSize, config.loc_dim), stddev=config.loc_std)
            net2Location = tf.stop_gradient(net2Location)

        return net1Location, net2Location