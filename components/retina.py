import tensorflow as tf
from components.glimpse import GlimpsNetwork
from components.location import LocationNetwork
from components.classifier import ClassifierNetwork
from components.preview import previewNetwork

class Retina():
    def __init__(self, images_ph):
        self.images_ph = images_ph
        self.location_network = LocationNetwork()  
        self.glimps_network = GlimpsNetwork(images_ph)
        self.classifierNetwork = ClassifierNetwork()

    def firstGlimpse(self):
        # init_location = tf.random_uniform((tf.shape(self.images_ph)[0], 2), minval=-1.0, maxval=1.0)
        location, guess = previewNetwork(self.images_ph)
        output = self.glimps_network(location, guess)

        return output

    def getNext(self, currentState):
        classifier = self.classifierNetwork(currentState)
        sample_coor = self.location_network(currentState, classifier.outputs)
        glimpse = self.glimps_network(sample_coor, classifier.outputs)

        return glimpse