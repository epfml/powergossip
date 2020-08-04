from abc import ABC, abstractmethod
from math import sqrt

import networkx
import numpy as np
from matplotlib import pyplot as plt


class Topology(ABC):
    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.diffusion_matrix = np.eye(self.num_workers)  # conservative default!
        self.name = "Unknown topology"

    def plot(self):
        """Show the diffusion matrix"""
        plt.matshow(self.diffusion_matrix)

    def neighbor_ranks(self, rank, include_self=False):
        """Lists all other ranks that worker `rank` is connected to."""
        assert rank >= 0 and rank < self.num_workers

        neighbors = np.where(self.diffusion_matrix[rank, :] > 0)[0]
        if not include_self:
            neighbors = neighbors[neighbors != rank]

        return neighbors

    def weight(self, sender_rank, receiver_rank):
        """Return the diffusion weight for a contribution sent from `sender_rank` to `receiver_rank`"""
        return self.diffusion_matrix[receiver_rank, sender_rank]

    def spectral_gap(self):
        eigen_values = sorted(np.linalg.eigvals(self.diffusion_matrix))
        assert np.abs(eigen_values[-1] - 1) < 1e-5
        return 1 - eigen_values[-2]

    @property
    def is_symmetric(self) -> bool:
        return np.allclose(self.diffusion_matrix, self.diffusion_matrix.T)

    @property
    def is_doubly_stochastic(self) -> bool:
        W = self.diffusion_matrix
        outgoing_messages_are_normalized = np.allclose(np.sum(W, axis=0), 1.0)
        incoming_messages_are_normalized = np.allclose(np.sum(W, axis=1), 1.0)
        return incoming_messages_are_normalized and outgoing_messages_are_normalized


class Ring(Topology):
    def __init__(self, num_workers):
        super().__init__(num_workers)

        eye = np.eye(num_workers)
        W = eye + np.roll(eye, 1, 1) + np.roll(eye, -1, 1)
        self.diffusion_matrix = W / np.sum(W, axis=1, keepdims=True)
        self.name = f"{num_workers}-ring"


class Torus(Topology):
    def __init__(self, num_workers):
        super().__init__(num_workers)

        dim1 = int(sqrt(num_workers))
        while dim1 * int(num_workers / dim1) != num_workers:
            dim1 -= 1
        dim2 = int(num_workers / dim1)

        # self-connections
        W = np.eye(num_workers) / 5

        for worker in range(num_workers):
            # Add edges along the primary ring
            W[worker, (worker + 1) % dim1 + (worker // dim1) * dim1] += 1 / 5
            W[worker, (worker - 1) % dim1 + (worker // dim1) * dim1] += 1 / 5

            # Add edges along the secondary
            W[worker, (worker + dim1) % num_workers] += 1 / 5
            W[worker, (worker - dim1) % num_workers] += 1 / 5

        self.diffusion_matrix = W
        self.name = f"{dim1}x{dim2}-torus"


class UnidirectionalRing(Topology):
    def __init__(self, num_workers):
        super().__init__(num_workers)

        eye = np.eye(num_workers)
        W = eye + np.roll(eye, 1, 1)
        self.diffusion_matrix = W / np.sum(W, axis=1, keepdims=True)
        self.name = f"{num_workers}-ring (uni-directional)"


class FullyConnected(Topology):
    def __init__(self, num_workers):
        super().__init__(num_workers)

        W = np.ones([num_workers, num_workers])
        self.diffusion_matrix = W / np.sum(W, axis=1, keepdims=True)
        self.name = f"fully-connected ({num_workers} workers)"


class SocialNetwork(Topology):
    """
    Code adapted from Koloskova and Lin et al. 2020.
    https://arxiv.org/pdf/1907.09356.pdf
    https://github.com/epfml/ChocoSGD/blob/master/dl_code/pcode/utils/topology.py
    """

    def __init__(self, num_workers):
        super().__init__(num_workers)
        assert num_workers == 32

        graph = networkx.davis_southern_women_graph()

        # get the mixing matrix.
        mixing_matrix = networkx.adjacency_matrix(graph).toarray().astype(np.float)

        degrees = mixing_matrix.sum(axis=1) + 1.0  # different from Koloskova et al.
        for node in np.argsort(degrees)[::-1]:
            mixing_matrix[:, node][mixing_matrix[:, node] == 1] = 1.0 / degrees[node]
            mixing_matrix[node, :][mixing_matrix[node, :] == 1] = 1.0 / degrees[node]
            mixing_matrix[node, node] = (
                1 - np.sum(mixing_matrix[node, :]) + mixing_matrix[node, node]
            )

        self.diffusion_matrix = mixing_matrix
        self.name = f"Davis southern women graph"


class ConnectedWattsStrogatz(Topology):
    def __init__(self, num_workers, k=3, p=0.9, seed=1):
        super().__init__(num_workers)

        graph = networkx.generators.random_graphs.connected_watts_strogatz_graph(
            num_workers, k, p, seed=seed
        )

        # get the mixing matrix.
        mixing_matrix = networkx.adjacency_matrix(graph).toarray().astype(np.float)

        degrees = mixing_matrix.sum(axis=1) + 1
        for node in np.argsort(degrees)[::-1]:
            mixing_matrix[:, node][mixing_matrix[:, node] == 1] = 1.0 / degrees[node]
            mixing_matrix[node, :][mixing_matrix[node, :] == 1] = 1.0 / degrees[node]
            mixing_matrix[node, node] = (
                1 - np.sum(mixing_matrix[node, :]) + mixing_matrix[node, node]
            )

        self.diffusion_matrix = mixing_matrix
        self.name = f"Connected Watts Strogatz graph (size {num_workers})"
