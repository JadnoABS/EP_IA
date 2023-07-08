import numpy as np
import pandas as pd

from mlp.neural_network import NeuralNetwork
from mlp.save_weights import SaveAndLoadMlp

import sys

if __name__ == "__main__":

    train = pd.read_csv("train.csv").values
    test = pd.read_csv("test.csv").values

    X_train = np.array([x_t[:63] for x_t in train])
    Y_train = np.array([y_t[63:] for y_t in train])
    X_test = np.array([x_t[:63] for x_t in test])
    Y_test = np.array([y_t[63:] for y_t in test])

    if len(sys.argv) > 1:
        if sys.argv[1] == 'train':
            print("Começando...")
            NN = NeuralNetwork(x_train=X_train, y_train=Y_train, n_layers=3,
                               rate=0.15, momentum=0.75,)
            SaveAndLoadMlp.save_mlp(NN, "initial")
            NN.train(X_train, Y_train)
            SaveAndLoadMlp.save_mlp(NN, "final")
        elif sys.argv[1] == 'test':
            print("Começando...")
            NN = SaveAndLoadMlp.read_mlp("final")
            NN.test(X_test, Y_test)
            for layer in NN.layers:
                for neuron in layer:
                    print(neuron.weights)

        else:
            print('Invalid Argument')
    else:
        print('You need to define a argument (train or test)')
