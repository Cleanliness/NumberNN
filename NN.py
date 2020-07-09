import numpy as np
from mnist import MNIST


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
        # vectorizing activation function and its derivative
        self.act_f = np.vectorize(sigmoid)
        self.act_f_prime = np.vectorize(sigmoid_prime)

        # set up neuron activation and random bias list
        self.activations = [np.array([0 for i in range(0, 784)])]
        self.biases = []
        for l in hidden_layers:
            self.activations.append(np.array([0 for i in range(0, l)]))
            self.biases.append(np.array([np.random.random_sample()*2 - 1 for i in range(0, l)]))
        self.activations.append(np.array([0 for i in range(0, 10)]))
        self.biases.append(np.array([np.random.random_sample()*2 - 1 for i in range(0, 10)]))

        # setting up neuron weight matrices w/ random weights
        self.weights = []
        for i in range(1, len(self.activations)):
            mat = []
            for r in range(0, len(self.activations[i])):
                row = np.array([np.random.random_sample()*2 - 1 for i in range(0, len(self.activations[i-1]))])
                mat.append(row)
            self.weights.append(np.array(mat))

        # setting up sum array with dummy values
        self.sum_arr = [1 for i in self.activations]

        # set up number to NN output conversion dict
        tmp = [0 for i in range(0, 9)]
        self.label_array = {0: tmp[:], 1: tmp[:], 2: tmp[:], 3: tmp[:], 4: tmp[:], 5: tmp[:], 6: tmp[:], 7: tmp[:],
                       8: tmp[:], 9: tmp[:]}
        for i in self.label_array:
            self.label_array[i].insert(i, 1)

    def GD(self, learning_rate=0.5, file_dir="training_data"):
        """Performs one gradient descent step on the NN given an image and label list.
        learning rate is a positive float representing how 'far' a step should be taken

        prints the cost of the NN for this step
        """

        print("started reading training data")
        data = MNIST(file_dir)
        images, labels = data.load_training()

        c = 0

        for i in range(0, len(images)):
            b = self.classify(images[i])
            tmp_c = (b - self.label_array[labels[i]])
            tmp_c = (tmp_c.dot(tmp_c))**2
            c += tmp_c
            self.backprop(self.label_array[labels[i]], learning_rate)

            print(tmp_c)

        print("training completed, average cost: " + str(c/60000))

    def classify(self, img):
        """
        classifies, i.e forward propagates the NN given a list, img

        img must be a 784 element list with elements in range of [0, 255]"""
        if len(img) != 784:
            return None

        arr_img = np.array(img)

        # set input layer to img, update sum array at first layer
        self.activations.pop(0)
        self.activations.insert(0, arr_img/255)

        self.sum_arr.pop(0)
        self.sum_arr.insert(0, arr_img)

        # update activations and sums for each layer
        for layer in range(1, len(self.activations)):
            self.activations.pop(layer)
            w = self.weights[layer - 1]
            a = self.activations[layer - 1]

            # calculate weighted sum of current layer
            l_sum = w.dot(a) + self.biases[layer-1]
            self.sum_arr.pop(layer)
            self.sum_arr.insert(layer, l_sum)

            # apply sigmoid to all sums and add it to the current activation matrix
            self.activations.insert(layer, self.act_f(l_sum))

        return self.activations[-1]

    def __split_batch(self, lst, size):
        """splits <lst> into a list containing numpy arrays with length of at least <size>
        and returns the new list"""

    def backprop(self, lbl, LR):
        """backpropagates the error of the NN when compared to a numpy array representing the expected output, lbl.
        updates all weights and biases of the NN for some learning rate, LR. label must correspond to the most
        recent image fed into the NN"""

        # getting error of output layer (hadamar product of grad_aC and sig_prime)
        # find partial derivative vector of cost WRT each activation(grad_aC), and sigmoid derivative vector (sig_prime)
        grad_aC = self.activations[-1] - lbl
        sig_prime = self.act_f_prime(self.sum_arr[-1])
        output_err = grad_aC * sig_prime
        err_list = [output_err]

        # backpropagating the error, find error of hidden layers
        for i in range(1, len(self.weights)):
            err = np.dot(np.transpose(self.weights[-i]), err_list[0]) * self.act_f_prime(self.sum_arr[-i - 1])
            err_list.insert(0, err)

        # update weights and biases given from error list
        for e in range(0, len(err_list)):
            # updating weights
            err_mat = np.array([err_list[e] for i in range(0, len(self.activations[e]))])
            act_mat = np.transpose(np.array([self.activations[e] for i in range(0, len(err_list[e]))]))
            w_grad = act_mat * err_mat
            old_w = self.weights.pop(e)
            new_w = np.transpose(np.transpose(old_w) - LR*w_grad)
            self.weights.insert(e, new_w)

            # updating biases
            old_b = self.biases.pop(e)
            new_b = old_b + LR*err_list[e]
            self.biases.insert(e, new_b)

    def load(self, fname="NN_save.txt"):
        """loads weights and biases from a text file"""
        f = open(fname, 'r')
        lines = f.readlines()

        self.__init__(eval(lines[1]))
        self.biases = eval(lines[3])
        self.weights = eval(lines[5])

    def save(self, fname="NN_save.txt"):
        """saves weights and biases to a text file.
        Formatting:

        a,b,c\n --> (a,b,c = # of neurons in corresponding hidden layers)

        [np.array(a), np.array(b), ... ]\n --> string representation of biases in NN where a,b are string reps of lists
                                                containing the biases of each layer(element of this list)

        [np.array([np.array(c,d)), np.array(d,e)], ... ]\n --> string rep of weights in NN, c,d are string reps of
                                                            lists containing the rows of each weight matrix
        """
        f = open(fname, 'w')

        # writing number of hidden layers
        f.write('----------------------hiddenLayers---------------------\n')
        f.write("[")
        for i in range(1, len(self.activations) - 1):
            f.write(str(len(self.activations[i])) + ",")
        f.write("]\n")

        # writing string rep of bias array
        f.write('----------------------biases---------------------\n')

        f.write("[")
        for i in range(0, len(self.biases)):
            s = np.array2string(self.biases[i], separator=",")
            f.write("np.array(" + np.array2string(self.biases[i], separator=",").replace("\n", "") + "),")
        f.write("]\n")

        f.write('----------------------weights---------------------\n')
        # writing string rep of weight array
        f.write("[")
        for wm in range(0, len(self.weights)):
            f.write("np.array([")
            for row in range(0, len(self.weights[wm])):
                f.write("np.array(" + np.array2string(self.weights[wm][row], separator=",").replace("\n", "") + "),")
            f.write("]),")
        f.write("]")

        f.close()


    def test(self, file_dir="training_data"):
        """tests the neural network given a directory, prints accuracy as a percentage"""
        print("loading testing data")
        test_data = MNIST(file_dir)
        img, lbl = test_data.load_testing()

        correct = 0
        for i in range(0, len(img)):
            self.classify(img[i])
            b = np.where(self.activations[-1] == max(self.activations[-1]))[0][0]
            c = lbl[i]
            if (np.where(self.activations[-1] == max(self.activations[-1]))[0][0]) == lbl[i]:
                correct += 1

        print(str((correct / len(img)) * 100) + " % accuracy")




def sigmoid(x):
    """activation function for the network"""
    return 1 / (1 + np.exp(-1 * x))


def sigmoid_prime(x):
    """derivative of the sigmoid function with respect to its input"""
    return sigmoid(x) * (1 - sigmoid(x))


