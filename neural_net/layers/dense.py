# dense.py
'''
Densely connected layer.
Inherits most of it's properties from the base layer, but adds instructions on
forward-and-backward passes, as well as calculate gradients and updateweights.
'''

# Things to fix
'''
'''

# Importing dependencies
import numpy as np
from .base import base
from ..necessities.serialization import array2json, json2array

# Code
class dense(base):
    def __init__(self, neurons, prevlayer, activation='tanh', cost='error'):
        # Base constructor
        super(dense, self).__init__(neurons, prevlayer, activation, cost)
        # Additional specifics constructor
        self.weights   = 2*np.random.random((prevlayer.neurons, neurons))-1
        self.bias      = 2*np.random.random((1,neurons))-1
        self.w_update  = np.zeros_like(self.weights)
        self.b_update  = np.zeros_like(self.bias)
    '''
    ----------------------------------------------------------------------------
    OVERWRITING BASE METHODS
    Overwriting empty basic methods inherited from base layer
    '''
    # Propagation
    def propagate_nonsequential(self):
        self.weighted_input = np.dot(self.prevlayer.value, self.weights) + self.bias
        self.value = self.activation(self.weighted_input)
    def propagate_sequential(self):
        self.propagate_nonsequential()
        self.pastvalues_remember()
    # Backpropagation
    def backpropagate_nonsequential_default(self):
        self.error = self.nextlayer.delta.dot(self.nextlayer.weights.T)
        self.calculate_gradients_nonsequential()
    def backpropagate_nonsequential_target(self, target):
        self.error = np.subtract(self.value, target)
        self.calculate_gradients_nonsequential()
    def backpropagate_sequential_default(self):
        self.pastvalues_recall()
        self.error = self.nextlayer.delta.dot(self.nextlayer.weights.T)
        self.calculate_gradients_sequential()
    def backpropagate_sequential_target(self, target):
        self.pastvalues_recall()
        self.error = np.subtract(self.value, target)
        self.calculate_gradients_sequential()
    # Calculate gradients
    def calculate_gradients_nonsequential(self):
        self.delta = np.multiply(self.error, self.act_prime(self.weighted_input))
    def calculate_gradients_sequential(self):
        self.delta = np.multiply(self.error, self.act_prime(self.weighted_input))
        self.w_update += self.prevlayer.pastvalue[-1].T.dot(self.delta)
        self.b_update += self.delta
    # Update weights
    def updateweights_nonsequential(self, alpha=1, clip=0.5):
        self.weights -= alpha*np.clip(self.prevlayer.value.T.dot(self.delta), -clip, clip)
        self.bias -= alpha*np.clip(np.mean(self.delta, axis=0), -clip, clip)
    def updateweights_sequential(self, alpha=1, clip=0.5):
        self.weights -= alpha*np.clip(self.w_update, -clip, clip)
        self.bias -= alpha*np.clip(self.b_update, -clip, clip)
    # Reset
    def reset_weightupdates(self):
        self.w_update = np.zeros_like(self.weights)
        self.b_update = np.zeros_like(self.bias)
    # Mutation
    def mutate(self, probability=0.1, mult=0.01):
        # Weights
        shape = (self.prevlayer.neurons, self.neurons)
        t = np.random.binomial(1, probability, size=shape)
        r = 1 - 2 * np.random.random(shape)
        self.weights += t * r * mult
        # Bias
        shape = (1, self.neurons)
        t = np.random.binomial(1, probability, size=shape)
        r = 1 - 2 * np.random.random(shape)
        self.bias += t * r * mult
    # Serialization
    def enjson_specs(self):
        lib = {'weights':{}, 'bias':{}}
        array2json(self.weights, lib['weights'])
        array2json(self.bias, lib['bias'])
        return lib
    def dejson_specs(self, lib):
        self.weights = json2array(lib['weights'])
        self.bias = json2array(lib['bias'])
        return self
    # Funny helpers
    def n_params(self):
        return self.neurons * (1 + self.prevlayer.neurons)
