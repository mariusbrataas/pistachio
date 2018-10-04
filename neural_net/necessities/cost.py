# costFunctions.py
'''
THESE HAVE NOT BEEN IMPLEMENTED YET!
All the most common cost functions
Use the function get_cost_function to pass cost_function name as text, and get
the cost function and it's prine back
'''

# Things to fix
'''
- crossentropy not working
'''

# Importing dependencies
import numpy as np

# Cost functions
def cost_error(value, target): return value-target
def cost_error_prime(value, target): return value-target
def cost_quadratic(value, target): return np.power(target-value, 2)/2
def cost_quadratic_prime(value, target): return (value-target)*2
def cost_crossentropy(value, target):
    return (-1/len(value))*np.sum((target*np.log(value))+(np.subtract(1,target)*np.log(np.subtract(1,value))), axis=0)
def cost_crossentropy_prime(value, target):
    return (-1/len(value))*np.sum((target/value)-np.subtract(1,target)/np.subtract(1,value),axis=0)
def getCostfunction(cost='error'):
    if 'error' in cost.lower(): return cost_error, cost_error_prime
    if 'quadratic' in cost.lower(): return cost_quadratic, cost_quadratic_prime
    if 'crossentropy' in cost.lower(): return cost_crossentropy, cost_crossentropy_prime
