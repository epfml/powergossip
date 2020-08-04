from abc import ABC, abstractmethod
from functools import partial

import numpy as np
import torch

from lib import bit2byte
from utils import pack, unpack


class TopK:
    def __init__(self, keep_ratio=None, rank=None):
        """
        Compress parameters by sending the coordinates with the largest absolute value
        The `rank` argument can be used to match the bits used by a low-rank method"""
        assert keep_ratio is not None or rank is not None
        self.keep_ratio = keep_ratio
        self.rank = rank

    def num_entries_for_param(self, param):
        keep_ratio = self.keep_ratio
        if keep_ratio is None and self.rank is not None:
            # Match the number of elements sent by a low-rank compressor
            m, n = param.view(param.shape[0], -1).shape
            keep_ratio = (m + n) / 2 / m / n * self.rank / 2  # one factor two is for indices
        return max(1, int(param.nelement() * keep_ratio))

    def compress(self, params):
        """Create a flat-packed message that can be sent to another worker"""
        values = []
        indices = []

        for tensor in params:
            top_size = self.num_entries_for_param(tensor)
            vals, positions = torch.topk(tensor.view(-1).abs(), top_size, sorted=False)
            del vals  # trying to get rid of memory leak
            indices.append(positions)
            values.append(tensor.view(-1)[positions].contiguous())

        # Pack compressed parameters together to reduce the number of communications
        value_buffer, shapes = pack(values)
        indices_buffer, _ = pack(indices)

        # Put this here to debug memory issues
        indices.clear()
        values.clear()

        return [value_buffer, indices_buffer], shapes

    def decompress_into(self, params, scalar_weight, messages, metadata):
        """
        Adds scaled, decompressed, parameters from `messages` to `params`.
        This modifies `params`.
        """
        shapes = metadata
        for param, values, indices in zip(
            params, unpack(messages[0], shapes), unpack(messages[1], shapes)
        ):
            indices = indices.long()
            param.data.view(-1)[indices] += scalar_weight * values
            del indices  # debugging memory issues


class RandomK(TopK):
    def compress(self, params):
        """Create a flat-packed message that can be sent to another worker"""
        values = []
        indices = []

        for tensor in params:
            sample_size = self.num_entries_for_param(tensor)
            positions = np.random.choice(tensor.nelement(), sample_size, replace=False)
            indices.append(torch.from_numpy(positions))
            values.append(tensor.view(-1)[positions].contiguous())

        # Pack compressed parameters together to reduce the number of communications
        value_buffer, shapes = pack(values)
        indices_buffer, _ = pack(indices)
        return [value_buffer, indices_buffer], shapes


class Identity:
    def compress(self, params):
        """Create a flat-packed message that can be sent to another worker"""
        buffer, _ = pack(params)
        return [buffer], None

    def decompress_into(self, params, scalar_weight, messages, metadata):
        """
        Adds scaled, decompressed, parameters from `messages` to `params`.
        This modifies `params`.
        """
        shapes = [p.shape for p in params]
        for param, values in zip(params, unpack(messages[0], shapes)):
            param[:].add_(scalar_weight, values)


class SVD:
    def __init__(self, rank):
        self.rank = rank

    def compress(self, params):
        """Create a flat-packed message that can be sent to another worker"""
        us = []
        vs = []
        for param in params:
            matrix = param.view(param.shape[0], -1)
            m, n = matrix.shape
            rnk = min(self.rank, m, n)

            if not matrix.is_cuda:
                try:
                    u, s, vT = np.linalg.svd(matrix)
                    v = vT.T
                    u, s, v = u[:, :rnk], s[:rnk], v[:, :rnk]
                except np.linalg.LinAlgError as e:
                    print("SVD failed ... using zeros")
                    u, s, v = (
                        np.zeros([matrix.shape[0], rnk], dtype=np.float32),
                        np.zeros([rnk], dtype=np.float32),
                        np.zeros([matrix.shape[1], rnk], dtype=np.float32),
                    )
                u = torch.from_numpy(u)
                s = torch.from_numpy(s)
                v = torch.from_numpy(v)
            else:
                u, s, v = torch.svd(matrix)
                u, s, v = u[:, :rnk], s[:rnk], v[:, :rnk]

            us.append((s * u).contiguous())
            vs.append(v.contiguous())
        buffer, shapes = pack(us + vs)

        return [buffer], shapes

    def decompress_into(self, params, scalar_weight, messages, metadata):
        """
        Adds scaled, decompressed, parameters from `messages` to `params`.
        This modifies `params`.
        """
        parts = unpack(messages[0], metadata)
        us = parts[: len(parts) // 2]
        vs = parts[len(parts) // 2 :]
        for param, u, v in zip(params, us, vs):
            param[:].addmm_(1, scalar_weight, u, v.t())


class SignAndNorm:
    def compress(self, params):
        """Create a flat-packed message that can be sent to another worker"""
        norms = torch.tensor([p.norm(p=1) for p in params], device=params[0].device)

        flat, _ = pack(params)

        bits, sign_size = bit2byte.packing(flat)

        return [bits, norms], sign_size

    def decompress_into(self, params, scalar_weight, messages, metadata):
        """
        Adds scaled, decompressed, parameters from `messages` to `params`.
        This modifies `params`.
        """
        sign_size = metadata

        signs = bit2byte.unpacking(messages[0], sign_size)
        norms = messages[1]

        shapes = [p.shape for p in params]

        for param, signs, norm in zip(params, unpack(signs, shapes), norms):
            param[:].add_(scalar_weight * norm / signs.nelement(), signs)
