from mlp.neural_network import NeuralNetwork
import pickle


class SaveAndLoadMlp:
    @staticmethod
    def save_mlp(neural_network: NeuralNetwork, file_name):
        with open(f'{file_name}.pkl', 'wb') as outp:
            pickle.dump(neural_network, outp, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def read_mlp(file_name) -> NeuralNetwork:
        with open(f'{file_name}.pkl', 'rb') as inp:
            neural_network = pickle.load(inp)
            return neural_network
