import tensorlayer as tl
import tensorflow as tf
import config


def imageGenerator(points):
    with tf.variable_scope('ImageGenerator', reuse = tf.AUTO_REUSE) as scope:
        # Glimps network upper part
        generator = tl.layers.InputLayer(tf.concat(points, 0), name='input_layer')
        generator = tl.layers.DenseLayer(generator, n_units = 196, name ='fc_1')
        generator = tl.layers.BatchNormLayer(generator, name='bn_1')
        generator = tf.nn.relu(generator.outputs, name='relu_1')

        # Glimps network lower part
        generator = tl.layers.InputLayer(generator, name='layer2')
        generator = tl.layers.DenseLayer(generator, n_units = 28*28, name ='fc_2')
        generator = tl.layers.BatchNormLayer(generator, name='bn_2')

    return generator.outputs


#  class ImageGenerator():
#     def __init__(self, batchSize):
#         self.images = tf.zeros([batchSize, 28*28], dtype=tf.float32)

#     def __call__(self, points):
#         with tf.variable_scope(self.scope or 'ImageGenerator', reuse = tf.AUTO_REUSE) as scope:
#             # Glimps network upper part
#             generator = tl.layers.InputLayer(tf.concat(points, 0), name='input_layer')
#             generator = tl.layers.DenseLayer(generator, n_units = 196, name ='fc_1')
#             generator = tl.layers.BatchNormLayer(generator, name='bn_1')
#             generator = tf.nn.relu(generator.outputs, name='relu_1')

#             # Glimps network lower part
#             generator = tl.layers.InputLayer(generator, name='layer2')
#             generator = tl.layers.DenseLayer(generator, n_units = 28*28, name ='fc_2')
#             generator = tl.layers.BatchNormLayer(generator, name='bn_2')

#             self.images = self.im
#         return generator.outputs