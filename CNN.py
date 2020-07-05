import numpy as np
from mnist import MNIST
import mnist

class NN:

    def __init__(self, hidden_layers):
        """format hidden_layers like this: [# of neurons in layer1, # in layer 2, ... ]"""

        # setting up neuron layers (input, hidden, output)
        self.layers = [[Neuron(sigmoid) for i in range(0, 784)]]

        first = 0
        for l_len in hidden_layers:
            self.layers.append([Neuron(sigmoid, self.layers[first]) for i in range(0, l_len)])
            first += 1
        self.layers.append([Neuron(sigmoid, self.layers[first]) for i in range(0, 10)])

        tmp = [0 for i in range(0, 9)]

        self.label_array = {0: tmp[:], 1: tmp[:], 2: tmp[:], 3: tmp[:], 4: tmp[:], 5: tmp[:], 6: tmp[:], 7: tmp[:],
                       8: tmp[:], 9: tmp[:]}

        for i in self.label_array:
            self.label_array[i].insert(i, 1)

    def train(self, file_dir="training_data"):
        data = MNIST(file_dir)
        images, labels = data.load_training()

        images = images[0:len(images) // 500]
        labels = labels[0:len(labels) // 500]
        tot_c = 0

        for i in range(0, len(images)):
            self.classify(images[i])
            c = np.linalg.norm(np.array([i.activation for i in self.layers[-1]]) - self.label_array[labels[i]]) ** 2
            tot_c += c
            #print(tot_c)

        cost = 1/(2*len(images)) * tot_c
        print(cost)

    def classify(self, img):
        """img must be a 784 element list with elements in range of [0, 255]"""
        if len(img) != 784:
            return None

        else:
            for i in range(0, len(self.layers[0])):
                self.layers[0][i].activation = img[i]/255

            for out_neuron in self.layers[-1]:
                out_neuron.fire()

            max_ind = -1
            for i in range(0, len(self.layers[-1])):
                if self.layers[-1][i].activation >= self.layers[-1][max_ind].activation:
                    max_ind = i

            return max_ind


    def load_weights(self):
        pass

    def get_out(self):
        return[n.activation for n in self.layers[-1]]

    def save(self):
        pass


class Neuron:
    def __init__(self, act_funct, prev_layer=None):
        if prev_layer is None:
            self.weights = []
        else:
            # initialize weights randomly
            self.weights = [np.random.random_sample()*2 - 1 for i in range(0, len(prev_layer))]

        self.activation = 0
        self.act_funct = act_funct
        self.prev = prev_layer

        # activation is sigmoid(sum of weights * activations of previous neurons)

    def fire(self):
        if self.prev is None:
            return self.activation

        else:
            tot_act = 0
            for i in range(0, len(self.prev)):
                tot_act += self.prev[i].fire() * self.weights[i]
            self.activation = sigmoid(tot_act)
            return tot_act


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


testNN = NN([10])
testNN.train()
