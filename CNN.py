import numpy as np
from mnist import MNIST
import mnist

class NN:

    def __init__(self, hidden_layers):
        """
        creates a new neural network/multilayer perceptron.

        format hidden_layers like this: [# of neurons in layer1, # in layer 2, ... ]

        neuron activation and bias list is set up like this:
        [a1, a2, ... an], where a1,... an are arrays containing neuron activations/biases at the nth layer
        ** bias matrix does not include first layer

        weight matrices are formatted like this, for some layer l:

        [w1a1, w2a1, ... w_n a_1]
        [w1a2, w2a2, ... w_n a_2]
        [           .           ]
        [           .           ]
        [           .           ]
        [w1 a_m, ....... w_n a_m]

        w_n a_m represents a weight connecting the the nth weight from the l-1th layer to the mth neuron
        """
        # set up neuron activation and bias list
        self.activations = [np.array([0 for i in range(0, 784)])]
        self.biases = []
        for l in hidden_layers:
            self.activations.append(np.array([0 for i in range(0, l)]))
            self.biases.append(np.array([np.random.random_sample()*2 - 1 for i in range(0, l)]))
        self.activations.append(np.array([0 for i in range(0, 10)]))
        self.biases.append(np.array([np.random.random_sample()*2 - 1 for i in range(0, 10)]))

        # setting up neuron weight matrices
        self.weights = []
        for i in range(1, len(self.activations)):
            mat = []
            for r in range(0, len(self.activations[i])):
                row = np.array([np.random.random_sample()*2 - 1 for i in range(0, len(self.activations[i-1]))])
                mat.append(row)
            self.weights.append(np.array(mat))

        # number to NN output conversion dict
        tmp = [0 for i in range(0, 9)]
        self.label_array = {0: tmp[:], 1: tmp[:], 2: tmp[:], 3: tmp[:], 4: tmp[:], 5: tmp[:], 6: tmp[:], 7: tmp[:],
                       8: tmp[:], 9: tmp[:]}
        for i in self.label_array:
            self.label_array[i].insert(i, 1)

    def GD(self, learning_rate = 0.01, file_dir="training_data"):
        """Performs one gradient descent step on the NN given an image and label list.
        learning rate is a positive float representing how 'far' a step should be taken

        prints the cost of the NN for this step
        """

        print("started reading training data")
        data = MNIST(file_dir)
        images, labels = data.load_training()

        d = 0
        for i in range(0, len(images)):
            b = self.classify(images[i])

            c = (b - self.label_array[labels[i]])
            c = (c.dot(c))**2
            d += 1

            print(c)

        print (str(d) + " operations")

    def classify(self, img):
        """
        classifies, i.e forward propagates the NN given a list, img

        img must be a 784 element list with elements in range of [0, 255]"""
        if len(img) != 784:
            return None

        # set input layer to img
        self.activations.pop(0)
        self.activations.insert(0, np.array([i/255 for i in img]))

        # update activations
        for layer in range(1, len(self.activations)):
            self.activations.pop(layer)
            w = self.weights[layer - 1]
            a = self.activations[layer - 1]

            new_act = np.array([sigmoid(i) for i in (w.dot(a) + self.biases[layer-1])])
            self.activations.insert(layer, new_act)

        # return np.where(self.activations[-1] == max(self.activations[-1]))[0][0]
        return self.activations[-1]

    def split_batch(self, lst, size):
        """splits <lst> into a list containing numpy arrays with length of at least <size>
        and returns the new list"""

    def load(self):
        """loads weights and biases from a text file"""
        pass

    def save(self):
        """saves weights and biases to a text file"""
        pass


def sigmoid(x):
    """activation function for the network"""
    return 1 / (1 + np.exp(-1 * x))


def sigmoid_prime(x):
    """derivative of the sigmoid function with respect to its input"""
    return sigmoid(x) * (1 - sigmoid(x))


testNN = NN([10,4])
testNN.GD()

b = 12
