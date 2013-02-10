import scipy
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.linalg import norm, pinv
from collections import defaultdict


class RBFN:

    def  __init__(self, n_centroids):
        self.n_centroids = n_centroids
        self.centroids = None
        self.stds = None
        self.weights = None

    def _discover_centroids(self, dataset_input):
        self.centroids, labels = kmeans2(dataset_input, self.n_centroids)
        while np.unique(labels).shape[0] != self.n_centroids:
            # print "Empty cluster found. Retrying kmeans.."
            self.centroids, labels = kmeans2(dataset_input, self.n_centroids)

        return (self.centroids, labels)

    def _calc_std_around_centroids(self, dataset_input, labels):
        """
        Args:
            centroids (1 x n_centroids): returned from kmeans
            labels (1 x n_samples): returned from kmeans

        Returns:
            stds: n_stds == n_centroids (there is 1-1 correspondence)
        """
        distances = defaultdict(lambda: [])
        self.stds = scipy.zeros(self.centroids.shape[0])

        for idx, centroid_idx in enumerate(labels):
            distance = norm(dataset_input[idx] - self.centroids[centroid_idx])
            distances[centroid_idx].append(distance)

        for centroid_idx, centroid in enumerate(self.centroids):
            self.stds[centroid_idx] = scipy.array(distances[centroid_idx]).std()

    def _calc_hidden_layer_output(self, dataset_input):
        """
        Returns:
            output (n_centroids+1 x n_samples): the output of the hidden layer.
        """
        output = scipy.zeros((
            self.centroids.shape[0]+1,
            dataset_input.shape[0]
        ))

        centroid_idx = 0
        for centroid, standard_deviation in zip(self.centroids, self.stds):
            output[centroid_idx] = self._rbf(dataset_input, centroid, standard_deviation)
            centroid_idx += 1

        # bias
        output[-1] = scipy.ones(dataset_input.shape[0])

        return output

    def _rbf(self, dataset_input, centroid, standard_deviation):
        """
        Returns:
            activation (n_samples x 1): activations of n_samples on one hidden unit
        """
        activation = (dataset_input - centroid)**2
        activation = scipy.sum(activation, axis=1)
        activation = activation / (2*standard_deviation**2)
        activation = scipy.exp(-activation)

        return activation

    def _init_rbf(self, dataset_input):
        _, labels = self._discover_centroids(dataset_input)
        self._calc_std_around_centroids(dataset_input, labels)

    def _calc_weights(self, dataset_input, dataset_output):
        """
        Returns:
            weights (n_centroids+1, n_categories)
        """
        hidden_layer_output = self._calc_hidden_layer_output(dataset_input)
        self.weights = scipy.dot(pinv(hidden_layer_output.T), dataset_output.T)

    def train(self, dataset_input, dataset_output):
        self._init_rbf(dataset_input)
        self._calc_weights(dataset_input, dataset_output)

    def predict(self, dataset_input):
        hidden_layer_output = self._calc_hidden_layer_output(dataset_input)
        return scipy.dot(self.weights.T, hidden_layer_output)
