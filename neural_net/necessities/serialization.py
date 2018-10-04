# tools.py
'''
Helpers for serialization.
'''

# Things to fix
'''
'''

# Importing dependencies
import numpy as np

# Code
def array2json(data, lib={}):
    lib['shape'] = np.shape(data)
    #print('np2json_array shape: ' + str(lib['shape']))
    lib['array'] = [float(element) for element in np.reshape(data, (np.product(lib['shape']),1))]
    return lib
def json2array(lib):
    return np.reshape(np.array(lib['array']), lib['shape'])
