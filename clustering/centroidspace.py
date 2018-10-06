# clusterspace.py
'''
Clustering algorithm.
'''

# Things to fix
'''
Nothing yet :)
'''

# Importing dependencies
import numpy as np

# Code
class centroid:
    def __init__(self, pos):
        self.pos = pos
        self.oldpos = []
        self.reset()
        self.labels = []
    def reset(self):
        self.assigned = []
        self.got_assigned = False
    def did_improve(self):
        return np.array_equal(self.oldpos, self.pos)
    def add(self, position):
        self.assigned.append(position)
        self.got_assigned = True
    def distance(self, positions):
        return np.sqrt(np.sum(np.square(np.subtract(self.pos, positions)), axis=1))
    def update(self):
        self.oldpos = self.pos.copy()
        if self.got_assigned: self.pos = np.mean(self.assigned, axis=0)
        self.reset()
    def addlabel(self, label):
        self.labels.append(label)
    def __str__(self):
        return 'Centroid at: %s' % self.pos

class centroidspace:
    def __init__(self, n_clusters=3, dims=2, init_positions=None):
        if init_positions != None:
            self.centroids = [centroid(position) for position in init_positions]
            self.n_clusters = len(self.centroids)
            self.dims = len(self.centroids[0].pos)
        else:
            self.n_clusters = n_clusters
            self.dims = dims
            self.centroids = [centroid([np.random.random()*0.00001 for dim in range(self.dims)]) for n in range(self.n_clusters)]
    def did_improve(self):
        for troid in self.centroids:
            if troid.did_improve(): return True
        return False
    def singlefit(self, positions, update=True):
        designations = np.argmin([troid.distance(positions) for troid in self.centroids], axis=0)
        for designation, position in zip(designations, positions): self.centroids[designation].add(position)
        if update:
            for troid in self.centroids: troid.update()
    def singlereduction(self, positions, epochs=50):
        for epoch in range(epochs):
            self.singlefit(positions)
            if not self.did_improve(): break
        self.singlefit(positions, update=False)
        self.centroids.pop(np.argmin([len(troid.assigned) for troid in self.centroids]))
        for troid in self.centroids: troid.update()
    def fit(self, positions, epochs=50):
        for epoch in range(epochs):
            self.singlefit(positions)
            if not self.did_improve():
                print('Stopped improving after %s epochs' % epoch)
                break
    def reductionfit(self, positions, epochs=50, min_centroids=7):
        for epoch in range(epochs):
            self.singlereduction(positions, epochs)
            if len(self.centroids) == min_centroids: break
    def predict(self, position, label=None):
        dists = [troid.distance(np.atleast_2d(position)) for troid in self.centroids]
        tmp = self.centroids[np.argmin(dists)]
        if label != None: tmp.addlabel(label)
        return tmp
    def disp(self):
        for troid in self.centroids:
            print(troid)
