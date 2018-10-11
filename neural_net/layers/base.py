# Importing dependencies
import numpy as np
from ..necessities.activation import getActivation
from ..necessities.cost import getCostfunction
from ..necessities.serialization import array2json, json2array

# get_layer
def get_layer(name='dense'):
    if name.lower() == 'base': return base
    if name.lower() == 'dense':
        from .dense import dense
        return dense
    if name.lower() == 'recurrent':
        from .recurrent import recurrent
        return recurrent

# Code
class base:
    def __init__(self, neurons=10, prevlayer=None, activation='tanh', cost='error'):
        # Basic
        self.neurons         = neurons
        self.value           = np.atleast_2d(np.zeros((1, self.neurons)))
        self.weighted_input  = np.atleast_2d(np.zeros((1, self.neurons)))
        self.error           = np.atleast_2d(np.zeros((1, self.neurons)))
        self.delta           = np.atleast_2d(np.zeros((1, self.neurons)))
        self.remember        = np.atleast_2d(np.zeros((1, self.neurons)))
        self.prevlayer       = prevlayer
        self.nextlayer       = None
        # Past
        self.pastvalue    = [np.zeros_like(self.value)]
        self.pastweighted = [np.zeros_like(self.value)]
        self.firstPastValue = True
        # Default method references
        self.propagate              = self.propagate_nonsequential
        self.propagate_sequence     = self.propagate_sequential
        self.backpropagate          = self.backpropagate_nonsequential_target
        self.backpropagate_sequence = self.backpropagate_sequential_target
        # Specific method references
        self.activation, self.act_prime = getActivation(activation)
        self.cost_function, self.cost_function_prime = getCostfunction(cost)
        # Constructor startup routines
        if self.prevlayer != None:
            # Update prevlayer method references
            self.prevlayer.nextlayer = self
            self.prevlayer.backpropagate = self.prevlayer.backpropagate_nonsequential_default
            self.prevlayer.backpropagate_sequence = self.prevlayer.backpropagate_sequential_default
            self.firstlayer = self.get_firstlayer()
        else:
            self.firstlayer = self
    '''
    ----------------------------------------------------------------------------
    METHODS TO OVERWRITE
    The following few methods are to be overwritten by classes extending
    this class. These methods are included here only for reference and to enable
    the base layer to perform it's own propagation
    '''
    # Propagation
    def propagate_nonsequential(self, inputs):
        self.weighted_input = np.atleast_2d(inputs)
        self.value = self.weighted_input
    def propagate_sequential(self, inputs):
        self.propagate_nonsequential(inputs)
        self.pastvalues_remember()
    # Backpropagation
    def backpropagate_nonsequential_default(self): return
    def backpropagate_nonsequential_target(self, target): return
    def backpropagate_sequential_default(self): self.pastvalues_recall()
    def backpropagate_sequential_target(self, target): self.pastvalues_recall()
    # Calculate gradients
    def calculate_gradients_nonsequential(self): return
    def calculate_gradients_sequential(self): return
    # Update weights
    def updateweights_nonsequential(self, alpha=1, clip=0.5): return
    def updateweights_sequential(self, alpha=1, clip=0.5): return
    # Reset
    def reset_weightupdates(self): return
    # Serialization
    def enjson_specs(self): return {}
    def dejson_specs(self, lib): return self
    '''
    ----------------------------------------------------------------------------
    METHODS TO INHERIT
    These methods will be inherited, but should NOT be overwritten.
    '''
    # Update weights
    def updateweights_all_nonsequential(self, alpha=1, clip=0.5):
        for layer in self.layers: layer.updateweights_nonsequential(alpha, clip)
    def updateweights_all_sequential(self, alpha=1, clip=0.5):
        for layer in self.layers: layer.updateweights_sequential(alpha, clip)
    # State
    def state_remember(self):
        self.remember = self.value.copy()
    def state_recall(self):
        self.pastvalue = [self.remember.copy()]
        self.value = self.remember.copy()
    # Pastvalues
    def pastvalues_remember(self):
        self.pastvalue.append(self.value.copy())
        self.pastweighted.append(self.weighted_input.copy())
        if self.firstPastValue:
            self.pastvalue[0] = np.zeros_like(self.pastvalue[1])
            self.pastweighted[0] = np.zeros_like(self.pastweighted[1])
            self.firstPastValue = False
    def pastvalues_recall(self):
        self.value = self.pastvalue[-1]
        self.pastvalue = self.pastvalue[:-1]
        self.weighted_input = self.pastweighted[-1]
        self.pastweighted = self.pastweighted[:-1]
    # Preparation
    def prep_batch(self, inputs=None):
        self.firstlayer.forward(inputs)
        self.firstlayer.reset_all()
    # Reset
    def reset_trainingmemory(self):
        self.pastvalue = [np.zeros_like(self.value)]
        self.pastweighted = [np.zeros_like(self.value)]
        self.firstPastValue = True
    def reset_memory(self):
        self.value           = np.atleast_2d(np.zeros((1, self.neurons)))
        self.weighted_input  = np.atleast_2d(np.zeros((1, self.neurons)))
        self.error           = np.atleast_2d(np.zeros((1, self.neurons)))
        self.delta           = np.atleast_2d(np.zeros((1, self.neurons)))
        self.remember        = np.atleast_2d(np.zeros((1, self.neurons)))
    def reset(self):
        self.reset_memory()
        self.reset_trainingmemory()
        self.reset_weightupdates()
    def reset_all(self):
        self.reset()
        for layer in self.layers: layer.reset()
    # Funny helpers
    def n_params(self):
        return 0
    def printspecs(self):
        toprint = 'Layer type: ' + str(self.__class__.__name__) + '\n'
        toprint += '- Neurons:    ' + str(self.neurons) + '\n'
        toprint += '- Activation: ' + str(self.activation.__name__) + '\n'
        print(toprint)
    # Getters
    def get_firstlayer(self):
        return self if self.prevlayer == None else self.prevlayer.get_firstlayer()
    def get_lastlayer(self):
        return self if self.nextlayer == None else self.nextlayer.get_lastlayer()
    # Mutation
    def mutate(self, probability=0.1, mult=0.01):
        return None
    # Serialization
    def enjson(self):
        baselib = {'value':{}, 'remember':{}}
        baselib['layertype'] = str(self.__class__.__name__)
        baselib['neurons'] = self.neurons
        array2json(self.value, baselib['value'])
        array2json(self.remember, baselib['remember'])
        baselib['activation'] = str(self.activation.__name__)
        baselib['cost_function'] = str(self.cost_function.__name__)
        speclib = self.enjson_specs()
        return {'basics':baselib, 'specifics':speclib}
    def enjson_specs(self):
        return {}
    # Deserialization
    def dejson(self, lib):
        self.value = json2array(lib['basics']['value'])
        self.remember = json2array(lib['basics']['remember'])
        self.dejson_specs(lib['specifics'])
        return self
    def dejson_specs(self, lib):
        return None
