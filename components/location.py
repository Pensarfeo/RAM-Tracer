import tensorlayer as tl
import tensorflow as tf
import config

class LocationSelector(object):
    def __call__(self, state, guess):
        with tf.variable_scope('location', reuse = tf.AUTO_REUSE):
            # Network structure

            state = tl.layers.InputLayer(state)
            state = tl.layers.DenseLayer(state, n_units = 128, name ='state_fc_1')
            state = tl.layers.BatchNormLayer(state, name='state_bn_1')
            state = tf.nn.relu(state.outputs, name='state_relu_1')

            guess = tl.layers.InputLayer(guess)
            guess = tl.layers.DenseLayer(guess, n_units = 128, name ='guess_fc_1')
            guess = tl.layers.BatchNormLayer(guess, name='guess_bn_1')
            guess = tf.nn.relu(guess.outputs, name='guess_relu_1')

            self.location_net = tl.layers.InputLayer(tf.concat([state, state], axis = 1))
            self.location_net = tl.layers.DenseLayer(self.location_net, n_units = 128 / 2, name='fc0')
            self.location_net = tl.layers.BatchNormLayer(self.location_net, name='bn_0')
            self.location_net = tf.nn.relu(self.location_net.outputs, name='relu_0')

            self.location_net = tl.layers.InputLayer(self.location_net, name='fc0-fc1')
            self.location_net = tl.layers.DenseLayer(self.location_net, n_units = config.loc_dim, name='fc1')                    
            self.location_net = tl.layers.BatchNormLayer(self.location_net, name='bn_1')
            self.location_net = tf.nn.relu(self.location_net.outputs, name='relu_1')    
            
            # self.location_net = tl.layers.InputLayer(self.location_net)
            # self.location_net = tl.layers.DenseLayer(self.location_net, n_units = config.loc_dim, name='fc2')

            # Add random
            self.mean = tf.stop_gradient(tf.clip_by_value(self.location_net, -1.0, 1.0))
            self.location = self.mean + tf.random_normal((tf.shape(state)[0], config.loc_dim), stddev=config.loc_std)
            self.location = tf.stop_gradient(self.location)
            return self.location