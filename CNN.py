import numpy as np


class NN:

    def __init__(self, hidden_layers):
        """format hidden_layers like this: [# of neurons in layer1, # in layer 2, ... ]"""

    def train(self):
        pass

    def classify(self, input):
        pass


class Neuron:
    def __init__(self, act_funct):
        self.weights = []
        self.activation = 0
        self.act_funct = act_funct

        # activation is sigmoid(weights * activations of previous neurons)


def sigmoid(x):
    return 1/(1+np.exp(-1*x))

