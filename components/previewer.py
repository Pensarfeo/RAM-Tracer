import tensorlayer as tl
import tensorflow as tf
import config

def previewer(images):
    with tf.variable_scope('getPreview'):
        preview_imgs = tf.reshape(images, [tf.shape(images)[0], 28, 28, 1], name='reshape_layer_1')
        preview_imgs = tf.image.resize_images(preview_imgs, [config.win_size, config.win_size])
        preview_imgs = tf.reshape(preview_imgs, [tf.shape(images)[0], config.win_size * config.win_size * 1])
        
        preview_net = tl.layers.InputLayer(preview_imgs, name='preview_layer')
        preview_net = tl.layers.DenseLayer(preview_net, n_units = 128, name ='prev_fc_1')
        preview_net = tl.layers.BatchNormLayer(preview_net, name='prev_bn_1')
        preview_net = tf.nn.relu(preview_net.outputs, name='prev_relu_1')

        with tf.variable_scope('firstLocation'):
            preview_net = tl.layers.InputLayer(preview_imgs, name='preview_layer')
            preview_net = tl.layers.DenseLayer(preview_net, n_units = 128, name ='prev_fc_1')
            preview_net = tl.layers.BatchNormLayer(preview_net, name='prev_bn_1')
            preview_net = tf.nn.relu(preview_net.outputs, name='prev_relu_1')

            preview_net = tl.layers.InputLayer(preview_net, name='preview_layer_2')
            preview_net = tl.layers.DenseLayer(preview_net, n_units = config.loc_dim, name ='prev_fc_2')
            preview_net = tl.layers.BatchNormLayer(preview_net, name='prev_bn_2')
            preview_net = tf.nn.relu(preview_net.outputs, name='prev_relu_2')

            location = tf.stop_gradient(tf.clip_by_value(preview_net, -1.0, 1.0))

    return location
