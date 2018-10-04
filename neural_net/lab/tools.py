# tools.py
'''
A collection of helpful tools that didn't really belong
anywhere.
'''

# Things to fix
'''
Nothing yet.
'''

# Importing dependencies
import numpy as np

# Sequential data preprocessor
def prepSequence(sequences):
    out = []
    for n in range(len(sequences[0])):
        out.append([seq[n] for seq in sequences])
    return np.array(out)

# Other tools
'''
For lack of a better place to put the softmax function...
'''
def softmax(values):
    values = np.exp(values)
    return values/np.sum(values)
