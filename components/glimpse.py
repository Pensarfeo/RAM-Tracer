import tensorlayer as tl
import tensorflow as tf
import config


class Glimpse(object):
    def __init__(self, images):
        self.images = images
        self.scope = None

    def getGlimpse(self, loc_tensor):
        with tf.variable_scope('getGlimpse'):
            self.glimps_imgs = tf.reshape(self.images, [tf.shape(self.images)[0], 28, 28, 1], name='reshape_layer_1')
            self.glimps_imgs = tf.image.extract_glimpse(self.glimps_imgs, [config.win_size, config.win_size], loc_tensor)
            self.glimps_imgs = tf.reshape(self.glimps_imgs, [tf.shape(loc_tensor)[0], config.win_size * config.win_size * 1])
        return self.glimps_imgs

    def __call__(self, loc_tensor):
        with tf.variable_scope(self.scope or 'Glimpse', reuse = tf.AUTO_REUSE) as scope:
            self.scope = self.scope or scope
            self.retina_imgs = self.getGlimpse(loc_tensor)

            # Glimps network upper part
            self.retina_net = tl.layers.InputLayer(self.retina_imgs, name='input_layer')
            self.retina_net = tl.layers.DenseLayer(self.retina_net, n_units = 128, name ='ind_fc_1')
            self.retina_net = tl.layers.BatchNormLayer(self.retina_net, name='ind_bn_1')
            self.retina_net = tf.nn.relu(self.retina_net.outputs, name='ind_relu_1')

            # Glimps network lower part
            self.location_net = tl.layers.InputLayer(loc_tensor, name='location_part_input_layer')
            self.location_net = tl.layers.DenseLayer(self.location_net, n_units = 128, name ='loc_fc_2')
            self.location_net = tl.layers.BatchNormLayer(self.location_net, name='loc_bn_2')
            self.location_net = tf.nn.relu(self.location_net.outputs, name='loc_relu_2')

            # Glimps network right part
            self.glimps_net = tf.concat([self.retina_imgs, self.location_net], axis=-1)       
            self.glimps_net = tl.layers.InputLayer(self.glimps_net, name='merge_input_layer')
            self.glimps_net = tl.layers.DenseLayer(self.glimps_net, n_units = 256, act = tf.nn.relu, name='fc_1')

        return self.glimps_net.outputs