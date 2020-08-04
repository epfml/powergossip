import os
from copy import deepcopy
from typing import Dict, Iterable, List

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader

from utils import DistributedSampler, MeanAccumulator

from . import cifar_architectures


class Batch:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class CifarTask:
    def __init__(
        self,
        device,
        timer,
        architecture,
        seed,
        use_decentralized_data,
        decentralized_data_shuffled_ratio,
    ):
        self._device = device
        self._timer = timer
        self._seed = seed
        self._architecture = architecture
        self._use_decentralized_data = use_decentralized_data

        self._train_set, self._test_set = self._create_dataset(
            data_root=os.path.join(os.getenv("DATA"), "data")
        )

        if self._use_decentralized_data:
            self._train_set = DecentralizedDataset(
                self._train_set, fraction_of_shuffled_data=decentralized_data_shuffled_ratio
            )
            my_rank = torch.distributed.get_rank()
            tally = np.bincount(self._train_set.targets, minlength=len(self._train_set.classes))
            print(f"Worker {my_rank} has {tally}")

        self._model = self._create_model()
        self._criterion = torch.nn.CrossEntropyLoss().to(self._device)

        self._epoch = 0  # Counts how many times train_iterator was called

        self.state = [parameter for parameter in self._model.parameters()]
        self.buffers = [buffer for buffer in self._model.buffers()]
        self.parameter_names = [name for (name, _) in self._model.named_parameters()]

    def train_iterator(self, batch_size: int) -> Iterable[Batch]:
        """Create a dataloader serving `Batch`es from the training dataset.
        Example:
            >>> for batch in task.train_iterator(batch_size=32):
            ...     batch_loss, gradients = task.batchLossAndGradient(batch)
        """
        if self._use_decentralized_data:
            train_loader = DataLoader(
                self._train_set,
                batch_size=batch_size,
                pin_memory=True,
                drop_last=True,
                shuffle=True,
                num_workers=1,
            )
        else:
            sampler = DistributedSampler(dataset=self._train_set, add_extra_samples=False)
            sampler.set_epoch(self._epoch)

            train_loader = DataLoader(
                self._train_set,
                batch_size=batch_size,
                sampler=sampler,
                pin_memory=True,
                drop_last=False,
                num_workers=1,
            )

        self._epoch += 1

        return BatchLoader(train_loader, self._device)

    def batch_loss(self, batch: Batch) -> (float, Dict[str, float]):
        """
        Evaluate the loss on a batch.
        If the model has batch normalization or dropout, this will run in training mode.
        Returns:
            - loss function (float)
            - bunch of metrics (dictionary)
        """
        with torch.no_grad():
            with self._timer("batch.forward", float(self._epoch)):
                prediction = self._model(batch.x)
                loss = self._criterion(prediction, batch.y)
            with self._timer("batch.evaluate", float(self._epoch)):
                metrics = self.evaluate_prediction(prediction, batch.y)
        return loss.item(), metrics

    def batch_loss_and_gradient(
        self, batch: Batch
    ) -> (float, List[torch.Tensor], Dict[str, float]):
        """
        Evaluate the loss and its gradients on a batch.
        If the model has batch normalization or dropout, this will run in training mode.
        Returns:
            - function value (float)
            - gradients (list of tensors in the same order as task.state())
            - bunch of metrics (dictionary)
        """
        self._zero_grad()
        with self._timer("batch.forward", float(self._epoch)):
            prediction = self._model(batch.x)
            assert not torch.isnan(prediction).any(), "diverged"
            f = self._criterion(prediction, batch.y)
        with self._timer("batch.backward", float(self._epoch)):
            f.backward()
        with self._timer("batch.evaluate", float(self._epoch)):
            metrics = self.evaluate_prediction(prediction, batch.y)
        df = [parameter.grad for parameter in self._model.parameters()]

        # classpred = torch.argmax(prediction, axis=-1)
        # bincounts = np.bincount(classpred.cpu(), minlength=10)
        # labelsbincount = np.bincount(batch.y.cpu(), minlength=10)
        # my_rank = torch.distributed.get_rank()
        # print(f"Prediction for {my_rank}: {bincounts}, labels {labelsbincount}")

        return f.detach(), df, metrics

    def evaluate_prediction(self, model_output, reference):
        """
        Compute a series of scalar loss values for a predicted batch and references
        """
        with torch.no_grad():
            _, top5 = model_output.topk(5)
            top1 = top5[:, 0]
            cross_entropy = self._criterion(model_output, reference)
            accuracy = top1.eq(reference).sum().float() / len(reference)
            # top5_accuracy = reference.unsqueeze(1).eq(top5).sum().float() / len(reference)
            return {
                "cross_entropy": cross_entropy.detach(),
                "accuracy": accuracy.detach(),
                # "top5_accuracy": top5_accuracy.detach(),
            }

    def test(self, state_dict=None) -> float:
        """
        Compute the average loss on the test set.
        """
        test_loader = BatchLoader(
            DataLoader(
                self._test_set, batch_size=250, num_workers=1, drop_last=False, pin_memory=True
            ),
            self._device,
        )

        if state_dict:
            test_model = self._create_test_model(state_dict)
        else:
            test_model = self._model
            test_model.eval()

        mean_metrics = MeanAccumulator()

        for batch in test_loader:
            with torch.no_grad():
                prediction = test_model(batch.x)
                metrics = self.evaluate_prediction(prediction, batch.y)
            mean_metrics.add(metrics)

        test_model.train()
        return mean_metrics.value()

    def state_dict(self):
        """Dictionary containing the model state (buffers + tensors)"""
        return self._model.state_dict()

    def _create_model(self):
        """Create a PyTorch module for the model"""
        with torch.random.fork_rng():
            torch.random.manual_seed(self._seed)
            model = getattr(cifar_architectures, self._architecture)()
        model.to(self._device)
        model.train()
        return model

    def _create_test_model(self, state_dict):
        test_model = deepcopy(self._model)
        test_model.load_state_dict(state_dict)
        test_model.eval()
        return test_model

    def _create_dataset(self, data_root="./data"):
        """Create train and test datasets"""
        dataset = torchvision.datasets.CIFAR10

        data_mean = (0.4914, 0.4822, 0.4465)
        data_stddev = (0.2023, 0.1994, 0.2010)

        transform_train = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomCrop((32, 32), 4),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(data_mean, data_stddev),
            ]
        )

        transform_test = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(data_mean, data_stddev),
            ]
        )

        training_set = dataset(root=data_root, train=True, download=True, transform=transform_train)
        test_set = dataset(root=data_root, train=False, download=True, transform=transform_test)

        return training_set, test_set

    def _zero_grad(self):
        self._model.zero_grad()


class BatchLoader:
    """
    Utility that transforms a DataLoader that is an iterable over (x, y) tuples
    into an iterable over Batch() tuples, where its contents are already moved
    to the selected device.
    """

    def __init__(self, dataloader, device):
        self._dataloader = dataloader
        self._device = device

    def __len__(self):
        return len(self._dataloader)

    def __iter__(self):
        for x, y in self._dataloader:
            x = x.to(self._device)
            y = y.to(self._device)
            yield Batch(x, y)


class DecentralizedDataset(torchvision.datasets.VisionDataset):
    """
    In a setting with multiple workers,
    this assigns a fixed portion of the dataset to each worker.
    In principle, they are sorted by label.
    The argument `fraction_of_shuffled_data` controls non-iidness.
    """

    def __init__(self, dataset, fraction_of_shuffled_data=0.0, rng=np.random.RandomState(0)):
        self.train = dataset.train
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.transform = dataset.transform
        self.target_transform = dataset.target_transform

        self.rng = rng
        self.fraction_of_shuffled_data = fraction_of_shuffled_data

        # Sort the dataset
        order = np.argsort(dataset.targets)
        self.all_data = dataset.data[order, :, :, :]
        self.all_targets = np.array(dataset.targets)[order]

        self.divide_data_across_workers()

    def divide_data_across_workers(self):
        """
        You can do this again to reshuffle a bit of the data if `fraction_of_shuffled_data` > 0
        """
        indices = np.arange(0, len(self.all_targets), dtype=np.long)

        if self.fraction_of_shuffled_data > 0:
            shuffle_indices = self.rng.choice(
                len(indices), size=int(self.fraction_of_shuffled_data * len(indices)), replace=False
            )
            shuffled_indices = shuffle_indices.copy()
            self.rng.shuffle(shuffled_indices)
            indices[shuffle_indices] = shuffled_indices

        indices_per_worker = np.array_split(indices, torch.distributed.get_world_size())
        my_indices = indices_per_worker[torch.distributed.get_rank()]

        self.targets = self.all_targets[my_indices]
        self.data = self.all_data[my_indices, :, :, :]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
