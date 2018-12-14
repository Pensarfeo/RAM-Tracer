import tensorflow as tf
import config

class Optimizer():
    def __init__(self, points, images):
        self.images = images
        self.points = points
    
    def getLosses(self, generatedImages, points):
        with tf.variable_scope('Losses'):
            sqrDif = tf.square(self.images - generatedImages)
            sqrDif = tf.reduce_sum(sqrDif)

            points = tf.convert_to_tensor(points)

            points1 = tf.slice(points, [0, 0, 0], [-1, -2 ,-1])
            points2 = tf.slice(points, [0, 1, 0], [-1, -1 ,-1])

            totPath = tf.reduce_sum(tf.square(points1 - points2))

            xyPoints = tf.transpose(points, [2,0,1])

            xPoints1 = tf.slice(xyPoints[0], [0, 0], [-2 ,-1])
            xPoints2 = tf.slice(xyPoints[0], [1, 0], [-1 ,-1])
            dXPoints = xPoints2 - xPoints1

            yPoints1 = tf.slice(xyPoints[1], [0, 0], [-2 ,-1])
            yPoints2 = tf.slice(xyPoints[1], [1, 0], [-1 ,-1])
            dYPoints = yPoints2 - yPoints1

            gradPoints = dYPoints/dXPoints

            gradPoints = tf.reduce_sum(gradPoints)

            a = 1
            b = 1
            c = 0

            loss = a * sqrDif + b * totPath + c * gradPoints

            # Hybric loss
            self.loss = loss
            self.var_list = tf.trainable_variables()
            self.grads = tf.gradients(self.loss, self.var_list)

    def setTrainer(self):
        with tf.variable_scope('Optimizer'):
            # Optimizer
            opt = tf.train.AdamOptimizer(0.0001)
            global_step = tf.get_variable('global_step', initializer=tf.constant(0), trainable=False)
            self.train_op = opt.apply_gradients(zip(self.grads, self.var_list), global_step=global_step)