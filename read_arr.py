import pickle
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression

def load_data():
    steps_arr, scores_arr = [], []
    with open('steps_arr', 'rb') as file:
        steps_arr = pickle.load(file)
    with open('scores_arr', 'rb') as file:
        scores_arr = pickle.load(file)
    return steps_arr, scores_arr

def load_model(filename = 'snake_nn.tflearn'):
    network = input_data(shape=[None, 5, 1], name='input')
    network = fully_connected(network, 25, activation='relu', name = 'hidden1')
    #network = fully_connected(network, 100, activation='relu')
    #network = fully_connected(network, 100, activation='relu')
    network = fully_connected(network, 1, activation='linear')
    network = regression(network, optimizer='adam', learning_rate=self.lr, loss='mean_square', name='target')
    model = tflearn.DNN(network)
    model.load(filename)
    return model

def get_weights():
    model = load_model()
    nn_vars = tflearn.get_layer_variables_by_name('hidden1')
    nn_weights = nn_vars[0]
    nn_biases = nn_vars[1]
    return nn_weights, nn_biases
