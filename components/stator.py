import tensorlayer as tl
import tensorflow as tf
import config

class Stator():
    def __init__(self, stateSize):
        self.lstm_cell = tf.nn.rnn_cell.LSTMCell(128, state_is_tuple=True)
        self.state = self.lstm_cell.zero_state(stateSize, tf.float32)

    def __call__(self, glimpse, prevPoint1, prevPoint2):
        with tf.variable_scope('stator', reuse = tf.AUTO_REUSE):

            glipmse = tl.layers.InputLayer(glimpse)
            glipmse = tl.layers.DenseLayer(glipmse, n_units = 128, name ='glipmse_fc_1')
            glipmse = tl.layers.BatchNormLayer(glipmse, name='glipmse_bn_1')
            glipmse = tf.nn.relu(glipmse.outputs, name='glipmse_relu_1')

            prevPoint1 = tl.layers.InputLayer(prevPoint1)
            prevPoint1 = tl.layers.DenseLayer(prevPoint1, n_units = 128, name ='prevPoint1_fc_1')
            prevPoint1 = tl.layers.BatchNormLayer(prevPoint1, name='prevPoint1_bn_1')
            prevPoint1 = tf.nn.relu(prevPoint1.outputs, name='prevPoint1_relu_1')

            prevPoint2 = tl.layers.InputLayer(prevPoint2)
            prevPoint2 = tl.layers.DenseLayer(prevPoint2, n_units = 128, name ='prevPoint2_fc_1')
            prevPoint2 = tl.layers.BatchNormLayer(prevPoint2, name='prevPoint2_bn_1')
            prevPoint2 = tf.nn.relu(prevPoint2.outputs, name='prevPoint2_relu_1')

            inputs = tf.concat([glipmse, prevPoint1, prevPoint2], axis=-1)       
            inputs = tl.layers.InputLayer(inputs, name='merge_input_layer')
            inputs = tl.layers.DenseLayer(inputs, n_units = 256, act = tf.nn.relu, name='fc_1')

            self.output, self.state = self.lstm_cell(inputs, self.state)
        return self.output