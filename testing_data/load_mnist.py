import csv, numpy as np

def load_mnist(name='mnist_test.csv'):
    rows = []
    with open(name) as file:
        for row in csv.reader(file, delimiter=','):
            rows.append((int(row[0]), np.reshape(np.array(row[1:], dtype='float'), (28,28))))
    return rows
