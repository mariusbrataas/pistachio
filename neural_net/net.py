# net.py
'''
Neural net.

This class adds functions to build, manage, train, and run inference on layers
with any kind of neurons. All layers inherit a lot of properties
from base_layer.py.
All layers are found in .layerstack
Use "add" or "quicklayers" to build new layers that are compatible with the
current layers in your neural net.


Example of use:
# Creating some inputs and targets
inputs = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
targets = [[0],[1],[0],[1],[0],[1],[0],[1]]

# Building a neural net
nn = neural_net(3)
nn.quicklayers('dense',[4,4,1],activation='sigmoid')

# Getting references to some layers because why not..
l0 = nn.first
l1 = nn.hidden[0]
l2 = nn.lastlayer

# Training our neural net
nn.fit(inputs,targets,epochs=1000,alpha=1,clip=5)

# Printing the results from the last training iteration
print(nn.lastlayer.value)

# Printing some specs
nn.disp()
'''

# Things to fix
'''
Nothing yet :)
'''

# Importing dependencies
import numpy as np, pickle, json
from .layers import base, dense, recurrent
from .layers.base import get_layer
from .necessities.tools import progressBar

# Softmax helper
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

# Code
class neural_net:
    def __init__(self, n_inputs):
        self.layerstack = [get_layer('base')(n_inputs)]
        self.first      = self.layerstack[0]
        self.hidden     = []
        self.last       = None
        self.errlog     = []
    # User
    def add(self, layertype='dense', neurons=10, activation='sigmoid', cost='error'):
        if len(self.layerstack) >= 2: self.hidden.append(self.layerstack[-1])
        self.layerstack.append(get_layer(layertype)(neurons, self.layerstack[-1], activation, cost))
        self.last = self.layerstack[-1]
        return self
    def quicklayers(self, layertype='dense', neurons=[10,10], activation='sigmoid', cost='error'):
        if not type(neurons) is list: neurons = [neurons]
        for neuron in neurons: self.add(layertype, neuron, activation, cost)
        return self
    def predict(self, inputs):
        self.propagate(inputs)
        return self.last.value
    def singlefit(self, inputs, targets, alpha=1, clip=0.25):
        self.propagate(inputs)                      # Forward pass
        self.backpropagate(targets)                 # Backward pass
        self.log_error()                            # Logging mean squared error
        self.updateweights(alpha=alpha, clip=clip)  # Updating weights
    def fit(self, inputs, targets, epochs=500, alpha=1, clip=0.25):
        for epoch in range(epochs):
            self.singlefit(inputs, targets, alpha, clip)
    def singlefit_sequential(self, inputs, targets, alpha=1, clip=0.25):
        self.pass_forward_sequence(inputs)                  # Forward pass
        self.pass_backward_sequence(targets)                # Backward pass
        self.log_error()                                    # Logging mean squared error
        self.updateweights_sequence(alpha=alpha, clip=clip) # Updating weights
    def fit_sequence(self, inputs, targets, epochs=500, alpha=1, clip=0.25, fliptargets=True, progress=True):
        if fliptargets: targets = list(reversed(targets))           # Flip targets to match respective inputs when backpropin'
        for epoch in range(epochs):
            self.singlefit_sequential(inputs, targets, alpha, clip) # Pass to singlefit
            if progress: progressBar(epoch, epochs-1, prefix='Training model', suffix='MSE = ' + str(np.mean(np.square(self.last.error))))
            self.reset() # Resetting
    def disp(self):
        print('\nNeural net structure\n #   Type         Size   Activation')
        for n in range(len(self.layerstack)):
            toprint = str(n).rjust(2) + ' | '
            toprint += (self.layerstack[n].__class__.__name__).ljust(10) + ' | '
            toprint += str(self.layerstack[n].neurons).rjust(4) + ' | '
            toprint += (self.layerstack[n].activation.__name__).rjust(4) if n != 0 else ''
            print(toprint)
    def disp_layerspecs(self):
        for n in range(len(self.layerstack)):
            print('Layer ' + str(n))
            self.layerstack[n].printspecs()
    def save(self, path):
        if not '.json' in path: path += '.json'
        with open(path, 'w') as outfile:
            json.dump(self.enjson(), outfile, indent=4)
    def reset(self):
        for layer in self.layerstack: layer.reset()
    # Propagate
    def propagate(self, inputs):
        self.first.propagate(inputs)
        for layer in self.hidden: layer.propagate()
        self.last.propagate()
    def propagate_sequence(self, inputs):
        self.first.propagate_sequence(inputs)
        for layer in self.hidden: layer.propagate_sequence()
        self.last.propagate_sequence()
    # Backpropagate
    def backpropagate(self, target):
        self.last.backpropagate_nonsequential_target(target)
        for layer in list(reversed(self.hidden)): layer.backpropagate_nonsequential_default()
        self.first.backpropagate_nonsequential_default()
    def backpropagate_sequence(self, target):
        self.last.backpropagate_sequential_target(target)
        for layer in list(reversed(self.hidden)): layer.backpropagate_sequential_default()
        self.first.backpropagate_sequential_default()
    # Updateweights
    def updateweights(self, alpha=1, clip=0.25):
        for layer in self.layerstack: layer.updateweights_nonsequential(alpha=alpha, clip=clip)
    def updateweights_sequence(self, alpha=1, clip=0.25):
        for layer in self.layerstack: layer.updateweights_sequential(alpha=alpha, clip=clip)
    # Pass
    def pass_forward_sequence(self, inputs):
        for inp in inputs: self.propagate_sequence(inp)
    def pass_backward_sequence(self, targets):
        for target in targets: self.backpropagate_sequence(target)
    # State
    def state_remember(self):
        for layer in self.layerstack: layer.remember_state()
    def state_recall(self):
        for layer in self.layerstack: layer.recall_state()
    # Predict
    def predict_action(self, inputs):
        self.propagate(inputs)
        return np.argmax(self.last.value)
    def predict_softmax(self, inputs):
        return softmax(self.predict(inputs))
    # Other
    def log_error(self):
        self.errlog.append(np.mean(np.square(self.last.error)))
    def copy(self):
        return neural_net.dejson(self.enjson())
    def mutate(self, probability=0.1, mult=0.01):
        for layer in self.layerstack:
            layer.mutate(probability, mult)
    # Serialization
    def enjson(self):
        lib = {'n_inputs': self.first.neurons}
        lib['firstlayer'] = self.first.enjson()
        lib['hiddenlayers'] = {}
        for layer in self.hidden:
            lib['hiddenlayers']['l' + str(1+len(lib['hiddenlayers']))] = layer.enjson()
        lib['lastlayer'] = self.last.enjson()
        return lib
    def save(self, path):
        if not '.json' in path: path += '.json'
        with open(path, 'w') as outfile:
            json.dump(self.enjson(), outfile, indent=4)
    @staticmethod
    def dejson(lib):
        nn = neural_net(lib['n_inputs'])
        nn.first.dejson(lib['firstlayer'])
        for l in lib['hiddenlayers']:
            tmplib = lib['hiddenlayers'][l]
            b = tmplib['basics']
            nn.add(b['layertype'], b['neurons'], b['activation'], b['cost_function'])
            nn.last.dejson(tmplib)
        b = lib['lastlayer']['basics']
        nn.add(b['layertype'], b['neurons'], b['activation'], b['cost_function'])
        nn.last.dejson(lib['lastlayer'])
        return nn
    def load(path):
        with open (path + '.json', 'rb') as tmp: loaded = json.load(tmp)
        return neural_net.dejson(loaded)
