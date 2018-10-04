# recurrent.py
'''
Multimodal connections.
Inherits most of it's properties from neural_layer, but adds instructions on
forwards-and-backwards pass, as well as updateweights.

Use add_multimodals to add connections to layers in front of this one.
Use add_recurrencies to add connections to layers behind this one.

A hint:
If connecting too many layers to this one you will probably run into
an overflow error in the activation. Solve this by clipping the values in
the activation, or simply NOT adding too many multimodal connections :D
'''

# Things to fix
'''
Nothing yet
'''

# Importing dependencies
import numpy as np
from .base_layer import base

# Code
class multimodal(base):
	def __init__(self, neurons, prevlayer, nextlayer=None, activation='tanh', cost='error'):
		super(multimodal_recurrent, self).__init__(neurons,prevlayer,nextlayer,activation,cost)
		self.weights        = 2*np.random.random((prevlayer.neurons, neurons))-1
		self.bias           = 2*np.random.random((1,neurons))-1
		self.multimodals	= []
		self.n_modals		= 0
		self.w_update       = [np.zeros_like(self.weights)]
		self.b_update       = [np.zeros_like(self.bias)]
		self.m_update 		= []
	def add_multimodals(self, layers):
		if not type(layers) is list: layers = [layers]
		for layer in layers:
			self.multimodals.append([layer, 2*np.random.random((layer.neurons, self.neurons))-1])
			self.n_modals += 1
    def propagate(self):
    	self.weighted_input = np.dot(self.prevlayer.value, self.weights) + self.bias
    	self.weighted_input += np.sum([np.dot(modal[0].value, modal[1]) for modal in self.multimodals], axis=0)
    	self.value = self.activation(self.weighted_input)
    def propagate_sequence(self):
    	self.propagate()
    	self.pastvalue.append(self.value.copy())
    	self.pastweighted.append(self.weighted_input.copy())
    def backprop(self):
    	self.error = self.nextlayer.delta.dot(self.nextlayer.weights.T)
    	self.calculate_gradients()
    def calculate_gradients(self):
		self.delta		= np.multiply(self.error, self.act_prime(self.weighted_input))
		self.w_update	= [self.prevlayer.pastvalue[-1].T.dot(self.delta)]
		self.m_update	= [modal[0].pastvalue[-1].T.dot(self.delta) for modal in self.multimodals]
		self.b_update	= [self.delta]
	def calculate_gradients_sequence(self):
		self.delta		= np.multiply(self.error, self.act_prime(self.weighted_input))
		self.w_update.append(self.prevlayer.pastvalue[-1].T.dot(self.delta))
		self.m_update.append([modal[0].pastvalue[-1].T.dot(self.delta) for modal in self.multimodals])
		self.b_update.append(self.delta)
	def reset_weightupdates(self):
		self.w_update	= [np.zeros_like(self.weights)]
		self.m_update	= [np.zeros_like(modal[1]) for modal in self.multimodals]
		self.b_update	= [np.zeros_like(self.bias)]
	def updateweights(self, alpha=1, clip=0.25):
		self.weights -= alpha*np.mean(np.clip(self.w_update, -clip, clip), axis=0)
		for n in range(self.n_modals):
			self.multimodals[n][1] -= alpha*np.mean(np.clip(self.m_update[n], .clip, clip), axis=0)
		self.bias -= alpha*np.mean(np.clip(self.b_update, -clip, clip), axis=0)
	def updateweights_sequence(self, alpha=1, clip=0.25):
		self.weights -= alpha*np.mean(np.clip(self.w_update, -clip, clip), axis=0)
		for n in range(self.n_modals):
			self.multimodals[n][1] -= alpha*np.mean(np.clip(self.m_update[n], .clip, clip), axis=0)
		self.bias -= alpha*np.mean(np.clip(self.b_update, -clip, clip), axis=0)
