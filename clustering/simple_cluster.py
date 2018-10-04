# simple_cluster.py
'''
Clustering.

This class file adds functions to manage and model cluster of data in
n dimensions (Euclidean space).


EXAMPLE:
# Let's say we've got some data
data = [[somedata1], [somedata2], ...]

# ... and let's say we got labels for our data as well.
labels = some_respective_labels

# Setting some hyperparameters
entry_centroids = 30
finish_centroids = 4

# Creating and fitting cluster
cluster = simplecluster(data, n_centroids=entry_centroids)
cluster.reduction_fit(data, entry=entry_centroids, finish=finish_centroids)

# Labels can now be grouped by their closest centroids!
grouped_labels = cluster.group_by_cluster(labels, data)
'''

# Things to fix
'''
Better descriptions, more tools..?
'''

# Importing dependencies
import numpy as np

# Other tools
def euclidean_distance(pos1, pos2, axis=None):
	return np.sqrt(np.sum(np.square(np.subtract(pos1, pos2)), axis=axis))

# Code
class simple_cluster:
	def __init__(self, data, n_centroids=2):
		self.n_centroids = n_centroids
		self.centroids = [data[np.random.randint(len(data))] for n in range(self.n_centroids)]
	def closest_centroid(self, position):
		return np.argmin(euclidean_distance(position, self.centroids, axis=1))
	def get_clusters(self, data):
		return np.argmin([euclidean_distance(centroid, data, axis=1) for centroid in self.centroids], axis=0)
	def group_by_cluster(self, labels, data):
		contents = [[] for n in range(self.n_centroids)]
		designations = self.get_clusters(data)
		for n in range(len(designations)):
			contents[designations[n]].append(labels[n].replace('.txt',''))
		return contents
	def remove_smallest_cluster(self, data):
		designations = self.get_clusters(data)
		counts = [0 for n in range(self.n_centroids)]
		for designation in designations: counts[designation] += 1
		self.centroids.pop(np.argmin(counts))
		self.n_centroids -= 1
	def reduction_fit(self, data, epochs=50, entry=15, finish=5, doprint=True):
		self.n_centroids = entry
		self.centroids = [data[np.random.randint(len(data))] for n in range(self.n_centroids)]
		if doprint: print('\nReductive fitting (' + str(entry-finish) + ' loops)')
		for n in range(entry-finish):
			if doprint: print('- Loop ' + str(n))
			self.fit(data, epochs=epochs)
			self.remove_smallest_cluster(data)
		self.fit(data, epochs=epochs)
	def fit(self, data, epochs=50):
		for epoch in range(epochs):
			done = True
			tmp_contents = [[] for n in range(self.n_centroids)]
			for position in data:
				tmp_contents[np.argmin(euclidean_distance(position, self.centroids))].append(position)
			for n in range(self.n_centroids):
				newpos = np.mean(tmp_contents[n], axis=0)
				if np.array_equal(newpos, self.centroids[n]):
					done = False
					self.centroids[n] = newpos
			if done: break	# Best positions found. No need to keep looking.
