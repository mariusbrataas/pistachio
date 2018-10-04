import numpy as np

def non_sequential():
    x = np.array([[0,0,0,0,0,0,0,0,0,0],
                [1,1,1,1,1,1,1,1,1,1],
                [1,0,1,0,1,0,1,0,1,0],
                [0,1,0,1,0,1,0,1,0,1]])
    y = np.array([[0,1,0,1,0,1,0,1,0,1],
                [0,0,1,1,0,0,1,1,0,0],
                [0,0,0,1,1,1,0,0,0,1],
                [0,0,0,0,1,1,0,0,0,0]])
    return x, y

def sequential():
    x = np.array([[[1 for n in range(10)],[0 for n in range(10)]] for i in range(10)])
    y = []
    for n in range(10):
        if n%2 == 0:
            s1 = [0 for i in range(10)]
        if n%2 == 1:
            s1 = [1 for i in range(10)]
        if n%4 <= 1:
            s2 = [0 for i in range(10)]
        if n%4 > 1:
            s2 = [1 for i in range(10)]
        y.append([s1,s2])
    return x, np.array(y)

def seq():
    x = [np.ones(10) for n in range(15)]
    y = [np.ones(10)*(n%2) for n in range(15)]
    return np.array(x), np.array(y)
