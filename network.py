import tensorlayer as tl
import tensorflow as tf

from components.previewer import previewer
from components.glimpse import Glimpse
from components.pathWriter import PathWriter
from components.stator import Stator
from components.location import LocationSelector
from components.imageGenerator import imageGenerator

import config

class MainLoop():
    def __init__(self, images, initLocation):
        images = images
        batchSize = tf.shape(images)[0]

        # set network pieces in main loop
        self.glimpse = Glimpse(images)
        self.pathWriter = PathWriter(batchSize)
        # misses image generator

        self.stator = Stator(batchSize)
        self.getNextGlimpseLocation = LocationSelector()
        self.locationTensor = initLocation
        self.pathPoints = []
        self.pathPointsArray = []

    def __call__(self):
        glimpseOutput = self.glimpse(self.locationTensor)
        p1, p2 = self.pathWriter(glimpseOutput)

        # gen image
        # we should ad a layer to mix pathoutput and glimpse results
        
        statorOutput = self.stator(glimpseOutput, p1, p2)
        self.locationTensor = self.getNextGlimpseLocation(statorOutput)
        self.pathPointsArray += [p1, p2]
        self.pathPoints += [p1[0], p1[1], p2[0], p2[1]]

        # append new image

def run(images_ph):    
    initLocation = previewer(images_ph)
    mainLoop = MainLoop(images_ph, initLocation)

    with tf.variable_scope('coreNetwork', reuse = tf.AUTO_REUSE):
        for i in range(0, 20):
            with tf.variable_scope('rnn'):
                mainLoop()

    images = imageGenerator(pathPoints)

    return mainLoop.pathPoints, images


