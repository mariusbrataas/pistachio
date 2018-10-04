# activation.py
'''
All the most common activations, along with their prime.
Use the function get_activation to pass the activation-type as text
and get the activation and prime back.
'''

# Things to fix
'''
Nothing yet.
'''

# Importing dependencies
import numpy as np

# Activations
def sigmoid(X, act_clip=100):
    X = np.clip(X, -act_clip, act_clip)
    return np.divide(1,np.add(1,np.exp(-X)))
def sigmoid_prime(X, act_clip=100):
    X = np.clip(X, -act_clip, act_clip)
    return sigmoid(X)*(1-sigmoid(X))
def ReLU(X):
    return np.maximum(X, 0, X)
def ReLU_prime(X):
    return 1 * (X > -1e-05)
def tanh(X, act_clip=100):
    X = np.clip(X, -act_clip, act_clip)
    return np.tanh(X)
def tanh_prime(X, act_clip=100):
    X = np.clip(X, -act_clip, act_clip)
    return 1/(np.square(np.cosh(X)))
def arctan(X):
    return np.arctan(X)
def arctan_prime(X):
    return 1/(np.power(X,2)+1.)
def sine(X):
    return np.sin(X)
def sine_prime(X):
    return np.cos(X)
def getActivation(act='tanh'):
    if act.lower() == 'sigmoid': return sigmoid, sigmoid_prime
    if act.lower() == 'relu': return ReLU, ReLU_prime
    if act.lower() == 'tanh': return tanh, tanh_prime
    if act.lower() == 'arctan': return arctan, arctan_prime
    if act.lower() == 'sine': return sine, sine_prime
