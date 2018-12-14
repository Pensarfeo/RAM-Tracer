import os
import tensorlayer as tl
import tensorflow as tf
import numpy as np
from tensorflow.contrib import distributions

from data import Prepare_dataset
from lib.saveData import DataSaver
from lib.timer import Timer
from pygit2 import Repository

import network
from components.optimizer import Optimizer
import config

# consts
EPHOCS = 25 * 2
TRAIN = True

runName = Repository('.').head.shorthand
modelSaveDir = os.path.join(os.getcwd(), 'output', runName, 'trainedModels')
modelSavePath = os.path.join(modelSaveDir, 'model.ckpt')
trainingString = 'training' if (TRAIN == True) else 'testing'
dataSavePath = os.path.join('output', runName, 'data', trainingString )
graphSavePath = os.path.join('output', runName, 'graph', trainingString )

print('graph location ======================================>')
print('tensorboard --logdir ', graphSavePath)
print('<====================================== graph location ')

if __name__ == '__main__':
    # Create placeholder
    images_ph = tf.placeholder(tf.float32, [None, 28 * 28])
    labels_ph = tf.placeholder(tf.int64, [None])

    # Create network
    mainLoop, imagesOutput = network.run(images_ph)

    # define loss
    optimizer = Optimizer(points, imagesOutput, labels_ph)

    # define optimizer

    

    saver = tf.train.Saver()
    roundDec = lambda x: "{0:.2f}".format(x)
    asdf
    if TRAIN:
        
        dataSaver = DataSaver('ephoch', 'iter', 'totLoss', 'fastConvergeEntropy', 'stableConvergeEntropy', 'reward', filename = dataSavePath)

        # Train
        with tf.Session() as sess:
            # save gaph
            tf.summary.FileWriter(graphSavePath).add_graph(sess.graph)

            mnist = Prepare_dataset(batch_size = config.batch_size)
            tf.global_variables_initializer().run()


            if os.path.isfile(modelSavePath + ".index"):
                saver.restore(sess, modelSavePath)

            timer = Timer(nsteps = (mnist.train_size // config.batch_size)*EPHOCS)

            if not os.path.exists(modelSaveDir):
                os.makedirs(modelSaveDir)

            for j in range(0, EPHOCS):
                for i in range(1, (mnist.train_size // config.batch_size)):
                    # images, labels = mnist.train.next_batch(config.batch_size)
                    images, labels = mnist(epoch = j)
                    images = np.tile(images, [config.M, 1])
                    labels = np.tile(labels, [config.M])

                    loss, reward, fcLoss, stLoss, _ = sess.run(
                        [
                            optimizer.loss,
                            optimizer.accuracy,
                            optimizer.fastConvergeEntropy,
                            optimizer.stableConvergeEntropy,
                            optimizer.train_op
                        ],
                        feed_dict = {
                            images_ph: images,
                            labels_ph: labels
                        }
                    )
                    
                    if i % (100) == 0:
                        dataSaver.add({
                            'ephoch': j
                            , 'iter': i
                            , 'totLoss': loss
                            , 'fastConvergeEntropy': fcLoss
                            , 'stableConvergeEntropy': stLoss
                            ,'reward': reward
                        })
                        print(
                            'ephoc: ', j,
                            '\titer: ', i,
                            '\tloss: ', roundDec(loss),
                            '\treward: ', roundDec(reward),
                            '\ttimeElapsed: ', timer.elapsed(step = (i + j * (mnist.train_size // config.batch_size))),
                            '\tfastConvergeEntropy :', roundDec(fcLoss),
                            '\tstableConvergeEntropy :', roundDec(stLoss),
                            '\tremaining: ', timer.left()
                        )
                if j % (5) == 0:
                    print('Tot Time Elapsed: ', timer.elpasedTot(), ' after ', j, ' steps')

                if ((j % (25) == 0) & (j != 0)):
                    print('------------------ Saving Session ------------------')
                    saver.save(sess, modelSavePath)
            print('------------------ Training Completed ------------------')
            print('Tot Time Elapsed ', timer.elpasedTot() )

    
            

    else:
        # --------------------------------------------------------------
        # test loop
        # --------------------------------------------------------------

        dataSaver = DataSaver('n', 'softmax', 'label', filename = dataSavePath, divider=',')
        
        with tf.Session() as sess:
            # save gaph
            # tf.summary.FileWriter('./temp/graph').add_graph(sess.graph)

            trainingBatchSize = config.batch_size
            mnist = Prepare_dataset(batch_size = trainingBatchSize)
            tf.global_variables_initializer().run()
            saver.restore(sess, modelSavePath)

            for i in range(1, (mnist.train_size // trainingBatchSize)):
                images, labels = mnist()

                _softmax = sess.run([softmax], feed_dict = {
                    images_ph: images,
                    labels_ph: labels
                })

                for j in range(0, trainingBatchSize):
                    dataSaver.add({
                            'n': i + j
                            ,'softmax': _softmax[0][j]
                            ,'label': labels[j]
                    })
                # print(i, i % 10)
                if i % (1000 // trainingBatchSize) == 0:
                    # dataSaver.add({
                    #     'n': i
                    #     ,'prediction': _labels_prediction
                    #     ,'label': labels
                    # })
                    print(
                        '\\n: ', i
                        # , '\tloss: ', _labels_prediction
                        # , '\treward: ', _reward_value
                    )