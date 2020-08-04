from collections import namedtuple
from math import sqrt

import numpy as np
import torch

from lib import bit2byte
from utils import num_bits, pack, unpack


def isend(*args, **kwargs):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return torch.distributed.isend(*args, **kwargs)


def recv(*args, **kwargs):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return torch.distributed.recv(*args, **kwargs)


class DPSGD:
    """
    Lian et al. Neurips 2017
    https://arxiv.org/pdf/1705.09056.pdf
    """

    def __init__(self, timer, gossip, params, update_fn, overlapping=True):
        self.timer = timer
        self.gossip = gossip
        self.gossip_state = gossip.init_state(params)
        self.params = params
        self.update_fn = update_fn
        self.overlapping = overlapping  # execute gradient and gossip in parallel on stale data

    def step(self, grad_fn):
        # To do: parallelize forward/backward + gossip
        with self.timer("forward_backward"):
            _, _, metrics = grad_fn()
            if not self.overlapping:
                self.update_fn()
        with self.timer("gossip"):
            gossiped_params, self.gossip_state = self.gossip.step(self.params, self.gossip_state)
        with self.timer("updates"):
            for param, gossiped in zip(self.params, gossiped_params):
                param.data = gossiped
            if self.overlapping:
                self.update_fn()
        return metrics

    @property
    def bits_sent(self):
        return self.gossip_state.bits_sent

    @property
    def messages_sent(self):
        return self.gossip_state.messages_sent


class SimpleGossip:
    """Implementation of x_{t+1} = W x_t"""

    State = namedtuple("State", ["bits_sent", "messages_sent"])

    def __init__(self, topology, diffusion_rate=1.0):
        self.topology = topology
        self.diffusion_rate = diffusion_rate

    def init_state(self, _):
        return self.State(bits_sent=0, messages_sent=0)

    def step(self, params, state):
        bits_sent = state.bits_sent
        messages_sent = state.messages_sent

        my_rank = torch.distributed.get_rank()

        # Send our values to the neighbors
        buffer, shapes = pack(params)
        send_request_handles = []
        for neighbor_rank in self.topology.neighbor_ranks(my_rank):
            handle = isend(buffer, neighbor_rank)
            bits_sent += num_bits(buffer)
            messages_sent += 1
            send_request_handles.append(handle)

        # Average with the neighbors
        own_weight = self.topology.weight(my_rank, my_rank)
        own_weight = 1.0 - (1.0 - own_weight) * self.diffusion_rate
        params = [param.data * own_weight for param in params]

        buffer = torch.empty_like(buffer)
        for neighbor_rank in self.topology.neighbor_ranks(my_rank):
            weight = self.topology.weight(my_rank, neighbor_rank)

            recv(buffer, neighbor_rank)

            for param, neighbor_param in zip(params, unpack(buffer, shapes)):
                param.data.add_(weight * self.diffusion_rate, neighbor_param)

        # Make sure all send requests finished
        for handle in send_request_handles:
            handle.wait()

        return params, self.State(bits_sent, messages_sent)


class OnlyOnLargeParameters:
    """
    Apply special a given gossip algorithm only on parameters with > 1 dimension, 
    use simple, uncompressed gossip for the rest.
    """

    State = namedtuple(
        "State", ["bits_sent", "messages_sent", "compressed_state", "uncompressed_state"]
    )

    def __init__(self, topology, gossip):
        self.gossip = gossip
        self.simple_gossip = SimpleGossip(topology)

    def init_state(self, params):
        return self.State(
            bits_sent=0,
            messages_sent=0,
            compressed_state=self.gossip.init_state([p for p in params if p.ndim > 1]),
            uncompressed_state=self.simple_gossip.init_state([p for p in params if p.ndim <= 1]),
        )

    def step(self, params, state):
        compressed_params, compressed_state = self.gossip.step(
            [p for p in params if p.ndim > 1], state.compressed_state
        )
        uncompressed_params, uncompressed_state = self.simple_gossip.step(
            [p for p in params if p.ndim <= 1], state.uncompressed_state
        )

        bits_sent = compressed_state.bits_sent + uncompressed_state.bits_sent
        messages_sent = compressed_state.messages_sent + uncompressed_state.messages_sent

        new_params = []
        i = 0
        j = 0
        for p in params:
            if p.ndim > 1:
                new_params.append(compressed_params[i])
                i += 1
            else:
                new_params.append(uncompressed_params[j])
                j += 1

        return (
            new_params,
            self.State(
                bits_sent=bits_sent,
                messages_sent=messages_sent,
                compressed_state=compressed_state,
                uncompressed_state=uncompressed_state,
            ),
        )


class AllReduce:
    """Ignores the topology and runs all-reduce as a baseline"""

    State = namedtuple("State", ["bits_sent", "messages_sent"])

    def __init__(self, topology):
        self.topology = topology

    def init_state(self, _):
        return self.State(bits_sent=0, messages_sent=0)

    def step(self, params, state):
        bits_sent = state.bits_sent
        messages_sent = state.messages_sent

        # Send our values to the neighbors
        buffer, shapes = pack(params)

        torch.distributed.all_reduce(buffer)
        bits_sent += num_bits(buffer)
        messages_sent += 1

        buffer /= torch.distributed.get_world_size()

        params = unpack(buffer, shapes)

        return params, self.State(bits_sent, messages_sent)


class ChocoGossip:
    """
    Koloskova et al. 2019.
    Decentralized Stochastic Optimization and Gossip Algorithms with Compressed Communication
    https://arxiv.org/pdf/1902.00340.pdf
    """

    # `local_x_hat` is x_hat for the current worker.
    # `neighbors_x_hat` is the weighted average of all neighbors. We don't need those separately.
    State = namedtuple("State", ["local_x_hat", "neighbors_x_hat", "bits_sent", "messages_sent"])

    def __init__(self, topology, diffusion_rate, compressor):
        self.topology = topology
        self.diffusion_rate = diffusion_rate
        self.compressor = compressor

    def init_state(self, params):
        return self.State(
            local_x_hat=[torch.zeros_like(p) for p in params],
            neighbors_x_hat=[torch.zeros_like(p) for p in params],
            bits_sent=0,
            messages_sent=0,
        )

    def step(self, params, state):
        local_x_hat, neighbors_x_hat, bits_sent, messages_sent = state
        my_rank = torch.distributed.get_rank()

        # Send updates on x-hat to the neighbors
        messages, compression_metadata = self.compressor.compress(
            [x - hat for x, hat in zip(params, local_x_hat)]
        )
        send_request_handles = []
        for neighbor_rank in self.topology.neighbor_ranks(my_rank):
            for message_number, buffer in enumerate(messages):
                handle = isend(buffer, neighbor_rank, tag=message_number)
                bits_sent += num_bits(buffer)
                messages_sent += 1
                send_request_handles.append(handle)

        # Update local_x_hat by adding the reconstructed difference between param and local_x_hat
        self.compressor.decompress_into(local_x_hat, 1.0, messages, compression_metadata)

        # Receive the message from neighbors, and update `neighbors_x_hat`
        receive_buffers = [torch.empty_like(msg) for msg in messages]
        for neighbor_rank in reversed(self.topology.neighbor_ranks(my_rank, include_self=True)):
            weight = self.topology.weight(my_rank, neighbor_rank)
            if my_rank == neighbor_rank:
                self.compressor.decompress_into(
                    neighbors_x_hat, weight, messages, compression_metadata
                )
            else:
                for tag, buffer in enumerate(receive_buffers):
                    recv(buffer, neighbor_rank, tag=tag)
                self.compressor.decompress_into(
                    neighbors_x_hat, weight, receive_buffers, compression_metadata
                )

        # Update the parameters
        params = [
            param.data + self.diffusion_rate * (Wxhat - xhat)
            for (param, xhat, Wxhat) in zip(params, local_x_hat, neighbors_x_hat)
        ]

        # Make sure all send requests finished
        for handle in send_request_handles:
            handle.wait()

        return params, self.State(local_x_hat, neighbors_x_hat, bits_sent, messages_sent)


class PowerGossip:
    State = namedtuple(
        "State", ["ps", "qs", "bits_sent", "messages_sent", "iteration_number", "rngs"]
    )

    def __init__(
        self,
        topology,
        rank=1,
        num_iterations=1,
        warm_start=True,
        synchronized_randomness=False,
        diffusion_rate=1.0,
        round_weights=False,
    ):
        self.topology = topology
        self.rank = rank
        self.num_iterations = num_iterations
        self.warm_start = warm_start
        self.synchronized_randomness = synchronized_randomness
        self.diffusion_rate = diffusion_rate
        self.round_weights = round_weights

    def init_state(self, params, seed=0):
        my_rank = torch.distributed.get_rank()
        ps = {}
        qs = {}

        rngs = {}
        for neighbor in self.topology.neighbor_ranks(my_rank):
            rngs[neighbor] = self._rng_for_neighbor(seed, neighbor)
            # Ensure that the p's and q's are consequtive in memory so we can quickly send them
            p_buffer, shapes = pack([self._init_p(param) for param in params])
            self.fill_with_random_values(p_buffer, rngs[neighbor])
            ps[neighbor] = {"list": unpack(p_buffer, shapes), "buffer": p_buffer}

            q_buffer, shapes = pack([self._init_q(param) for param in params])
            self.fill_with_random_values(q_buffer, rngs[neighbor])
            qs[neighbor] = {"list": unpack(q_buffer, shapes), "buffer": q_buffer}

        return self.State(ps=ps, qs=qs, iteration_number=0, bits_sent=0, messages_sent=0, rngs=rngs)

    def _rng_for_neighbor(self, seed, neighbor_rank):
        rank = torch.distributed.get_rank()
        a, b = min(rank, neighbor_rank), max(rank, neighbor_rank)
        if self.synchronized_randomness:
            shared_seed = seed
        else:
            shared_seed = int(str(seed) + str(a) + str(b))
        return np.random.RandomState(shared_seed)

    def _init_p(self, param):
        m, n = param.view(param.shape[0], -1).shape
        rnk = min(m, n, self.rank)
        return torch.empty([m, rnk], device=param.device)

    def _init_q(self, param):
        m, n = param.view(param.shape[0], -1).shape
        rnk = min(m, n, self.rank)
        return torch.empty([n, rnk], device=param.device)

    def fill_with_random_values(self, tensor, rng):
        seed = rng.randint(1_000_000_000)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            tensor.data[:] = torch.randn(*tensor.shape, device=tensor.device)

    def step(self, params, state):
        ps, qs, bits_sent, messages_sent, iteration_number, rngs = state

        my_rank = torch.distributed.get_rank()

        params = [param.data.clone() for param in params]

        for iteration in range(iteration_number, iteration_number + self.num_iterations):

            if iteration == 0:
                # there has been no gradient update yet, so no disagreement between the neighbors
                break

            # Switch between left and right matrix multiplications
            if iteration % 2 == 1:
                ps, qs = qs, ps
                p_and_q_are_swapped = True
                transpose_if_even = lambda m: m
            else:
                p_and_q_are_swapped = False
                transpose_if_even = lambda m: m.t()

            if not self.warm_start:
                for neighbor in self.topology.neighbor_ranks(my_rank):
                    self.fill_with_random_values(ps[neighbor]["buffer"], rngs[neighbor])

            request_handles = []
            for neighbor in self.topology.neighbor_ranks(my_rank):
                # Do a local matrix multiplication
                for tensor, p, q in zip(params, ps[neighbor]["list"], qs[neighbor]["list"]):
                    if self.round_weights:
                        assert p.shape[1] == 1
                        p[:] = p[:].sign() / sqrt(p.shape[0])
                    else:
                        orthogonalize(p)
                    matrix = tensor.view(tensor.shape[0], -1)
                    torch.matmul(transpose_if_even(matrix), p, out=q[:])

                # Send the flattened vector with results to the neighbors
                handle = isend(qs[neighbor]["buffer"], neighbor)
                bits_sent += num_bits(qs[neighbor]["buffer"])
                messages_sent += 1
                request_handles.append(handle)

            any_neighbor = self.topology.neighbor_ranks(my_rank)[0]
            recv_buffer = torch.empty_like(qs[any_neighbor]["buffer"])
            for handle, neighbor in zip(request_handles, self.topology.neighbor_ranks(my_rank)):
                # Recieve their results
                recv(recv_buffer, neighbor)
                handle.wait()

                # Store the outcome of the matrix multiplication (x_i - x_j)p, where i > j
                if my_rank > neighbor:
                    qs[neighbor]["buffer"].sub_(recv_buffer)
                else:
                    qs[neighbor]["buffer"][:] = recv_buffer - qs[neighbor]["buffer"]

            if p_and_q_are_swapped:
                # Swap back
                ps, qs = qs, ps

            for neighbor in self.topology.neighbor_ranks(my_rank):
                weight = self.topology.weight(my_rank, neighbor)
                for tensor, p, q in zip(params, ps[neighbor]["list"], qs[neighbor]["list"]):
                    sign = -1 if my_rank > neighbor else 1
                    tensor.data.add_(
                        sign * weight * self.diffusion_rate, (p @ q.t()).view(*tensor.shape)
                    )

        return (
            params,
            self.State(
                ps, qs, bits_sent, messages_sent, iteration_number + self.num_iterations, rngs
            ),
        )


class DeepSqueezeGossip:
    """
    Tang et al. 2019
    https://arxiv.org/pdf/1907.07346.pdf
    Add a gradient update before performing gossip to obtains DeepSqueeze (SGD) as in the paper
    """

    State = namedtuple("State", ["delta", "bits_sent", "messages_sent"])

    def __init__(self, topology, diffusion_rate, compressor):
        self.topology = topology
        self.diffusion_rate = diffusion_rate
        self.compressor = compressor

    def init_state(self, params):
        return self.State(delta=[torch.zeros_like(p) for p in params], bits_sent=0, messages_sent=0)

    def step(self, params, state):
        delta, bits_sent, messages_sent = state
        my_rank = torch.distributed.get_rank()

        params = [param.data.clone() for param in params]

        # Compute v and compress
        v = [param + d for param, d in zip(params, delta)]
        messages, compression_metadata = self.compressor.compress(v)

        # Compute a new delta = v - C(v)
        delta = [v_entry.clone() for v_entry in v]
        self.compressor.decompress_into(delta, -1.0, messages, compression_metadata)

        send_request_handles = []
        for neighbor_rank in self.topology.neighbor_ranks(my_rank):
            for message_number, buffer in enumerate(messages):
                handle = isend(buffer, neighbor_rank, tag=message_number)
                bits_sent += num_bits(buffer)
                messages_sent += 1
                send_request_handles.append(handle)

        # Receive the message from neighbors and update the params
        own_weight = self.topology.weight(my_rank, my_rank)
        self.compressor.decompress_into(
            params, -self.diffusion_rate * (1 - own_weight), messages, compression_metadata
        )
        receive_buffers = [torch.empty_like(msg) for msg in messages]
        for neighbor_rank in self.topology.neighbor_ranks(my_rank):
            weight = self.topology.weight(my_rank, neighbor_rank)
            for tag, buffer in enumerate(receive_buffers):
                recv(buffer, neighbor_rank, tag=tag)
            self.compressor.decompress_into(
                params, self.diffusion_rate * weight, receive_buffers, compression_metadata
            )

        # Make sure all send requests finished
        for handle in send_request_handles:
            handle.wait()

        return (params, self.State(delta, bits_sent, messages_sent))


class MoniquaGossip:
    """
    Lu et al. 2020
    https://arxiv.org/pdf/2002.11787.pdf
    2-bit Moniqua with stochastic rounding and shared randomness
    """

    State = namedtuple("State", ["bits_sent", "messages_sent"])

    def __init__(self, topology, diffusion_rate, theta):
        self.topology = topology
        self.diffusion_rate = diffusion_rate
        self.theta = theta
        self.delta = 1 / 3
        self.Btheta = 2 * theta / (1 - 2 * self.delta)
        self.rng = np.random.RandomState(0)  # for shared randomness

    def init_state(self, params):
        return self.State(bits_sent=0, messages_sent=0)

    def modulo(self, x, a):
        """This is zero when x is an integer multiple of a"""
        return torch.fmod(x + torch.sign(x) * a / 2, a) - torch.sign(x) * a / 2

    def stochastic_rounding(self, x, noise):
        """Assumes x is between 0 and 1"""
        return torch.floor(x / self.delta + noise) * self.delta

    def step(self, params, state):
        bits_sent, messages_sent = state

        # Pack all parameters in one flat vector `p` to speed up the computation
        p, original_shapes = pack(params)

        seed = self.rng.randint(1_000_000_000)
        noise = torch.rand_like(p)

        q = self.stochastic_rounding(self.modulo(p / self.Btheta, 1) + 0.5, noise) - 0.5
        xhat = q * self.Btheta - self.modulo(p, self.Btheta)  # left out + x_{k_i}

        # Send compressed messages to the neighbors
        my_rank = torch.distributed.get_rank()
        send_handles = []
        for neighbor_rank in self.topology.neighbor_ranks(my_rank):
            handle = isend(q, neighbor_rank)
            bits_sent += q.nelement() * 2
            messages_sent += 1
            send_handles.append(handle)

        # Receive messages and update the parameter p
        recv_buffer = torch.empty_like(q)
        for neighbor_rank in self.topology.neighbor_ranks(my_rank):
            recv(recv_buffer, neighbor_rank)
            weight = self.topology.weight(my_rank, neighbor_rank)
            p.add_(
                self.diffusion_rate * weight,
                self.modulo(recv_buffer * self.Btheta - p, self.Btheta) - xhat,
            )

        # Make sure all sends are finished
        for handle in send_handles:
            handle.wait()

        # Unpack the parameters back into a list form
        params = unpack(p, original_shapes)
        return params, self.State(bits_sent, messages_sent)


@torch.jit.script
def orthogonalize(matrix, eps=torch.tensor(1e-8)):
    n, m = matrix.shape
    for i in range(m):
        # Normalize the i'th column
        col = matrix[:, i : i + 1]
        col /= torch.sqrt(torch.sum(col ** 2)) + eps
        # Project it on the rest and remove it
        if i + 1 < m:
            rest = matrix[:, i + 1 :]
            # rest -= torch.matmul(col.t(), rest) * col
            rest -= torch.sum(col * rest, dim=0) * col
