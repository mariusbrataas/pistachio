# dropout.py
'''
The most common dropout functions
'''

# Things to fix
'''
- Check out shuffle
'''

# Importing dependencies
import numpy as np

# Dropout
def dropout_empty(layerValueRef, p=0.1):
    return
def dropout_hinton(layerValueRef, p=0.1):
    layerValueRef = np.multiply(layerValueRef, np.random.binomial(1, (1.0-p), size=np.shape(layerValueRef)) * (1.0/(1.0-p)))
def dropout_shuffle(layerValueRef, p=0.1):
    if np.random.random() < p: np.random.shuffle(layerValueRef)
