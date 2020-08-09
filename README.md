# NumberNN
Neural network/Multilayer perceptron implementation in python with a GUI. Trains using gradient descent. Classifies numbers given a 28x28 pixel image. Saves/loads weights and biases to a text file. Can reach ~93% accuracy on MNIST test data.

# Requirements
- python 3
- pyglet (for GUI)
- numpy
- python-mnist

# Usage
- Run num_painter.py if you want to use the GUI, automatically creates a NN and loads weights and biases contained in NN_save.txt
- NN.py contains the neural network implementation, can test, train, and classify by itself if you don't want to use the GUI/don't have pyglet installed
- Create an instance of the NN object and call GD() in NN.py to train it given MNIST data in the training_data folder
- Call Classify() to test/feed forward the NN
- Call load('filename') on an NN object to load a NN from a text file, call save('filename') to save one. (default filename = NN_save.txt)
- Sample NN configurations are saved in the 'models' folder, filenames are their accuracy on MNIST testing data

# Screenshots

![No Input](https://i.postimg.cc/pXwhYvTj/Nonepic.png)
![Writing Something Down](https://i.postimg.cc/K8tK0WkH/three.png)
![Classifying](https://i.postimg.cc/Gpr4Vjvh/output.png)
