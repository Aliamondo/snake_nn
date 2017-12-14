import pickle
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
from tensorflow.python.framework import ops

def get_stats():
    steps_arr, scores_arr = [], []
    with open('steps_arr', 'rb') as file:
        steps_arr = pickle.load(file)
    with open('scores_arr', 'rb') as file:
        scores_arr = pickle.load(file)
    return steps_arr, scores_arr

def get_data(filename = 'snake_nn.tflearn'):
    ops.reset_default_graph()
    input_nn = input_data(shape=[None, 5, 1], name='input')
    hidden1 = fully_connected(input_nn, 25, activation='relu')
    #hidden1 = fully_connected(input_nn, 100, activation='relu')
    #hidden2 = fully_connected(hidden1, 100, activation='relu')
    #hidden3 = fully_connected(hidden2, 100, activation='relu')
    output = fully_connected(hidden1, 1, activation='linear')
    network = regression(output, optimizer='adam', learning_rate=0.01, loss='mean_square', name='target')
    model = tflearn.DNN(network)
    model.load(filename)

    #nn_vars = tflearn.get_layer_variables_by_name('hidden1')
    nn_weights = hidden1.W
    nn_biases = hidden1.b
    w = model.get_weights(hidden1.W)
    with model.session.as_default():
        b = tflearn.variables.get_value(hidden1.b)
    return w, b
