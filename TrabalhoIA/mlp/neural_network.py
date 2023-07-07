import numpy as np
from math import ceil

from mlp.neurons import HiddenNeuron, OutNeuron
from mlp.Debugger import Debugger


class NeuralNetwork:
    layers: list = []

    def __init__(self, n_layers, rate,
                 momentum, x_train=np.array([]), y_train=np.array([]),
                 x_test=np.array([]), y_test=np.array([]),
                 weights=[]):

        if len(weights) == 0:
            self.construct_layers(n_inputs=x_train.shape[1],
                                  n_outputs=y_train.shape[1],
                                  n_layers=n_layers)
        else:
            self.construct_layers(weights=weights)

        self.rate = rate
        self.momentum = momentum

    def construct_layers(self, n_inputs, n_outputs, n_layers, weights=[]):
        self.layers = []
        if len(weights) == 0:
            for index, layer in enumerate(weights):
                layer = []
                for neuron in weights:
                    if index == len(weights) - 1:
                        new_neuron = OutNeuron(weights=neuron)
                    else:
                        new_neuron = HiddenNeuron(weights=neuron)
                    layer.append(new_neuron)
                self.layers.append(layer)

        n_hidden = ceil((n_inputs / 3) * 2)
        self.layers.append(
            [HiddenNeuron(next_layer_size=n_inputs) for _ in range(n_hidden)])

        for i in range(n_layers - 2):
            self.layers.append([HiddenNeuron(next_layer_size=n_hidden) for _ in
                                range(n_hidden)])

        self.layers.append(
            [OutNeuron(next_layer_size=n_hidden) for _ in range(n_outputs)])

    def propagate(self, inputs: list):
        next_input = inputs
        for index, layer in enumerate(self.layers):
            input_arr = np.array(next_input)
            next_input = []
            for neuron in layer:
                output = neuron.activate(input_arr)
                next_input.append(output)

        return next_input

    def calculate_initial_error(_, result, expected):
        err = 0
        for index, x in enumerate(expected):
            err += 0.5 * ((x - result[index]) ** 2)

        return err

    def calculate_errors(self, result, expected, input_array):
        total_err = self.calculate_initial_error(result, expected)

        ik = self.get_outputs_from_layer(-2)
        for i, outNeuron in enumerate(self.layers[-1]):
            error = expected[i] - result[i]
            outNeuron.calculate_deltas(self.rate, self.momentum, error, ik)

        for index in range(len(self.layers) - 2, -1, -1):
            weights = np.transpose(self.get_weights_from_layer(index + 1))
            error_info = self.get_errors_from_layer(index + 1)
            total_weights = self.get_total_weight_from_layer(index + 1)

            outputs = []

            if index == 0:
                outputs = input_array.tolist()
                outputs.append(1)
            else:
                outputs = self.get_outputs_from_layer(index - 1)

            for pos, hiddenNeuron in enumerate(self.layers[index]):
                hiddenNeuron.calculate_deltas(self.rate, self.momentum,
                                              error_info, weights[pos], outputs)

        for layer in self.layers:
            for neuron in layer:
                neuron.update_weight()

        return total_err

    def get_outputs_from_layer(self, index):
        outputs = [x.output for x in self.layers[index]]
        outputs.append(1)
        return outputs

    def get_weights_from_layer(self, index):
        return [x.weights for x in self.layers[index]]

    def get_errors_from_layer(self, index):
        return [x.error_info for x in self.layers[index]]

    def get_total_weight_from_layer(self, index):
        result = 0
        for weight in self.get_weights_from_layer(index):
            result += weight
        return result

    def train(self, input_array, expected, x_test, y_test):
        epoch = 0
        stop = False
        max_precision_epochs = 0
        while not stop:
            if epoch == 1000:
                stop = True

            training_errors = 0
            for index in range(len(input_array)):
                result = self.propagate(input_array[index])
                # print(self.convert_to_alphabet(result), self.convert_to_alphabet(expected[index]))
                self.calculate_errors(result, expected[index],
                                      input_array[index])
                if np.array(result).argmax() != np.array(expected[index]).argmax():
                    training_errors += 1
            epoch += 1
            # Debugger.pause()
            print('\n', end='')
            print("Epoch: ", epoch)
            precision = self.test(input_array, expected)
            if precision >= 1:
                max_precision_epochs += 1
            else:
                max_precision_epochs = 0
            if max_precision_epochs >= 5:
                stop = True

    def test(self, x_test, y_test):
        err = 0
        total = 0.0
        for index, received in enumerate(x_test):
            results = self.propagate(x_test[index])
            received = np.array(results).argmax()
            expected = np.array(y_test[index]).argmax()
            if received != expected:
                err += 1
            total += 1

        precision = 1 - (err / total)

        print("Number of errors: ", err)
        print("precision: ", precision)
        return precision

    def convert_to_alphabet(self, results):
        max_index = np.array(results).argmax()
        match max_index:
            case 0:
                return 'A'
            case 1:
                return 'B'
            case 2:
                return 'C'
            case 3:
                return 'D'
            case 4:
                return 'E'
            case 5:
                return 'J'
            case 6:
                return 'K'
