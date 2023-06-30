import unittest
from random import random
import numpy as np

from neurons import BaseNeuron, Perceptron


class NeuronTest(unittest.TestCase):
    def test_should_calc(self):
        base_neuron = BaseNeuron(3)
        input_arr = np.array([random() - 0.5 for i in range(3)])
        base_neuron.receive(input_arr)


if __name__ == '__main__':
    unittest.main()
