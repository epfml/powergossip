import math
import time
import json
from contextlib import contextmanager
from copy import deepcopy
from io import StringIO

import numpy as np
import torch


NS = 1.0 / 1_000_000_000  # 1[ns] in [s]


def pack(tensors):
    """Packs a list of tensors into one buffer for sending to other workers"""
    buffer = torch.cat([t.view(-1) for t in tensors])  # copies
    shapes = [tensor.shape for tensor in tensors]
    return buffer, shapes


def unpack(buffer, shapes):
    """Provides pointers to tensors of original `shapes` in a flat-packed buffer."""
    idx = 0
    entries = []
    for tensor_shape in shapes:
        end = idx + tensor_shape.numel()
        entries.append(buffer[idx:end].view(size=tensor_shape))
        idx = end

    return entries


def num_bits(tensor):
    return tensor.nelement() * 8 * tensor.element_size()


class MeanAccumulator:
    def __init__(self, update_weight=1):
        self.average = None
        self.counter = 0
        self.update_weight = update_weight

    def value(self):
        if isinstance(self.average, dict):
            return {k: v.value() for k, v in self.average.items()}
        elif isinstance(self.average, list):
            return [v.value() for v in self.average]
        else:
            return self.average

    def reduce(self):
        """Reduce over workers"""
        if not torch.distributed.is_available() or torch.distributed.get_world_size() == 1:
            # Skip this if there is only one worker
            return

        if isinstance(self.average, dict):
            for key in sorted(self.average.keys()):
                self.average[key].reduce()
        elif isinstance(self.average, list):
            for avg in self.average:
                avg.reduce()
        else:
            device = "cuda" if torch.distributed.get_backend() == "nccl" else "cpu"
            total_count = torch.tensor(self.counter, dtype=torch.float32, device=device)
            handle_tc = torch.distributed.all_reduce(total_count, async_op=True)

            # Average * count
            if isinstance(self.average, torch.Tensor):
                multiplied = self.average.clone()
            else:
                multiplied = torch.tensor(self.average, dtype=torch.float32, device=device)
            multiplied.mul_(self.counter)
            handle_mul = torch.distributed.all_reduce(multiplied, async_op=True)

            handle_tc.wait()
            handle_mul.wait()

            self.counter = total_count.item()

            if isinstance(self.average, torch.Tensor):
                self.average.data = multiplied / total_count
            else:
                self.average = (multiplied / total_count).item()

    def add(self, value, weight=1.0):
        """Add a value to the average"""
        self.counter += weight
        if self.average is None:
            self._init(value, weight)
        else:
            if isinstance(self.average, dict):
                for k, v in value.items():
                    self.average[k].add(v, weight)
            elif isinstance(self.average, list):
                for avg, new_value in zip(self.average, value):
                    avg.add(new_value, weight)
            else:
                self._update(value, weight)

    def _update(self, value, weight):
        alpha = float(self.update_weight * weight) / float(self.counter + self.update_weight - 1)
        if isinstance(self.average, torch.Tensor):
            self.average.mul_(1.0 - alpha)
            self.average.add_(alpha, value)
        elif isinstance(self.average, float):
            self.average *= 1.0 - alpha
            self.average += alpha * value
        else:
            raise ValueError("Unknown type")

    def _init(self, value, weight):
        if isinstance(value, dict):
            self.average = {}
            for key in value:
                self.average[key] = MeanAccumulator()
                self.average[key].add(value[key], weight)
        elif isinstance(value, list):
            self.average = []
            for v in value:
                acc = MeanAccumulator()
                acc.add(value[key], weight)
                self.average.append(acc)
        else:
            self.average = deepcopy(value)

    def reset(self):
        self.average = None
        self.counter = 0


class Timer:
    """
    Timer for PyTorch code
    Comes in the form of a contextmanager:
    Example:
    >>> timer = Timer()
    ... for i in range(10):
    ...     with timer("expensive operation"):
    ...         x = torch.randn(100)
    ... print(timer.summary())
    """

    def __init__(self, verbosity_level=1, log_fn=None, skip_first=False):
        self.verbosity_level = verbosity_level
        self.log_fn = log_fn if log_fn is not None else self._default_log_fn
        self.skip_first = skip_first

        self.epoch = 0

        self.reset()

    def reset(self):
        """Reset the timer"""
        self.means = {}  # Mean time per label
        self.m2 = {}  # Squared distance from the mean
        self.totals = {}  # Total time per label
        self.first_time = {}  # First occurrence of a label (start time)
        self.last_time = {}  # Last occurence of a label (end time)
        self.call_counts = {}  # Number of times a label occurred

    @contextmanager
    def __call__(self, label, epoch=-1.0, verbosity=1):
        if epoch == -1.0:
            epoch = self.epoch

        # Don't measure this if the verbosity level is too high
        if verbosity > self.verbosity_level:
            yield
            return

        # Measure the time
        self._cuda_sync()
        start = time.time_ns() * NS
        yield
        self._cuda_sync()
        end = time.time_ns() * NS

        # Update first and last occurrence of this label
        if not label in self.first_time:
            self.first_time[label] = start
        self.last_time[label] = end

        # Update the totals and call counts
        if not label in self.means and self.skip_first:
            self.means[label] = 0.0
            self.totals[label] = 0.0
            del self.first_time[label]
            self.call_counts[label] = 0
        elif not label in self.means and not self.skip_first:
            self.call_counts[label] = 1
            self.means[label] = end - start
            self.totals[label] = end - start
            self.m2[label] = 0.0
        else:
            self.call_counts[label] += 1
            new_value = end - start
            self.totals[label] += new_value
            delta = new_value - self.means[label]
            self.means[label] += delta / self.call_counts[label]
            delta2 = new_value - self.means[label]
            self.m2[label] += delta * delta2

        if self.call_counts[label] > 0:
            # We will reduce the probability of logging a timing linearly with the number of times
            # we have seen it.
            # It will always be recorded in the totals, though
            if np.random.rand() < 1 / self.call_counts[label]:
                self.log_fn(
                    "timer", {"epoch": float(epoch), "value": end - start}, {"event": label}
                )

    def transcript(self):
        t = []
        for label in sorted(self.totals):
            t.append(
                {
                    "event": label,
                    "instances": self.call_counts[label],
                    "mean": self.means[label],
                    "std": (
                        self.m2[label] / (self.call_counts[label] - 1)
                        if self.call_counts[label] > 1
                        else 0
                    ),
                    "total": self.totals[label],
                }
            )
        return t

    def summary(self):
        """
        Return a summary in string-form of all the timings recorded so far
        """
        with StringIO() as buffer:
            print("--- Timer summary -----------------------------------------------", file=buffer)
            print("  Event                          |  Count | Average time |  Frac.", file=buffer)
            for event_label in sorted(self.totals):
                total = self.totals[event_label]
                count = self.call_counts[event_label]
                if count == 0:
                    continue
                avg_duration = total / count
                total_runtime = self.last_time[event_label] - self.first_time[event_label]
                runtime_percentage = 100 * total / total_runtime
                print(
                    f"- {event_label:30s} | {count:6d} | {avg_duration:11.5f}s | {runtime_percentage:5.1f}%",
                    file=buffer,
                )
            print("-----------------------------------------------------------------", file=buffer)
            return buffer.getvalue()

    def save_summary(self, json_file_path):
        data = {}
        for event_label in sorted(self.totals):
            total = self.totals[event_label]
            count = self.call_counts[event_label]
            if count == 0:
                continue
            avg_duration = total / count
            data[event_label] = {
                "event": event_label,
                "average_duration": avg_duration,
                "n_events": count,
                "total_time": total,
            }

        with open(json_file_path, "w") as fp:
            json.dump(data, fp)

    def _cuda_sync(self):
        """Finish all asynchronous GPU computations to get correct timings"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _default_log_fn(self, _, values, tags):
        label = tags["event"]
        epoch = values["epoch"]
        duration = values["value"]
        print(f"Timer: {label:30s} @ {epoch:4.1f} - {duration:8.5f}s")


class DistributedSampler(torch.utils.data.distributed.Sampler):
    """
    This is a copy of torch.utils.data.distributed.DistributedSampler (28 March 2019)
    with the option to turn off adding extra samples to divide the work evenly.
    """

    def __init__(self, dataset, add_extra_samples=True):
        self._dataset = dataset
        if torch.distributed.is_available():
            self._num_replicas = torch.distributed.get_world_size()
            self._rank = torch.distributed.get_rank()
        else:
            self._num_replicas = 1
            self._rank = 0
        self._add_extra_samples = add_extra_samples
        self._epoch = 0

        if add_extra_samples:
            self._num_samples = int(math.ceil(len(self._dataset) * 1.0 / self._num_replicas))
            self._total_size = self._num_samples * self._num_replicas
        else:
            self._total_size = len(self._dataset)
            num_samples = self._total_size // self._num_replicas
            rest = self._total_size - num_samples * self._num_replicas
            if self._rank < rest:
                num_samples += 1
            self._num_samples = num_samples

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self._epoch)
        indices = torch.randperm(len(self._dataset), generator=g).tolist()

        # add extra samples to make it evenly divisible
        if self._add_extra_samples:
            indices += indices[: (self._total_size - len(indices))]
        assert len(indices) == self._total_size

        # subsample
        indices = indices[self._rank : self._total_size : self._num_replicas]
        assert len(indices) == self._num_samples

        # This wasn't there before, which seems a bug?
        # Is the user supposed to do this?
        self.set_epoch(self._epoch + 1)

        return iter(indices)

    def __len__(self):
        return self._num_samples

    def set_epoch(self, epoch):
        self._epoch = epoch
