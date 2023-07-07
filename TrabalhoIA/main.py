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
    # X_train = np.array([
    # [1, 1],
    # [1, -1],
    # [-1, 1],
    # [-1, -1]
    # ])
    # Y_train = np.array([
    # [-1],
    # [1],
    # [1],
    # [-1]
    # ])
    # X_test = np.array([
    # [1, 1],
    # [1, -1],
    # [-1, 1],
    # [-1, -1]
    # ])
    # Y_test = np.array([
    # [-1],
    # [1],
    # [1],
    # [-1]
    # ])
    # print(X_train[0])
    # print(Y_train[0])
    # print(X_train.shape)

    if len(sys.argv) > 1:
        if sys.argv[1] == 'train':
            print("Começando...")
            NN = NeuralNetwork(x_train=X_train, y_train=Y_train, x_test=X_test,
                               y_test=Y_test, n_layers=4,
                               rate=0.1, momentum=0.5)
            SaveAndLoadMlp.save_mlp(NN, "initial")
            NN.train(X_train, Y_train, X_test, Y_test)
            SaveAndLoadMlp.save_mlp(NN, "final")
        elif sys.argv[1] == 'test':
            print("Começando...")
            NN = SaveAndLoadMlp.read_mlp("final")
            NN.test(X_test, Y_test)
        else:
            print('Invalid Argument')
    else:
        print('You need to define a argument (train or test)')
