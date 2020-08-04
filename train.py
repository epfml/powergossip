#!/usr/bin/env python3

import datetime
import os
import re
import time
from copy import deepcopy
from math import sqrt

import numpy as np
import torch

import algorithms
import compressors
import tasks
import topologies
from utils import Timer, MeanAccumulator, pack, unpack

config = dict(
    task="Cifar",
    task_architecture="ResNet18",
    batch_size=128,  # per worker
    weight_decay=0.0001,
    num_epochs=300,
    learning_rate=0.1,
    momentum=0.9,
    topology="ring",
    lr_schedule_milestones=[(150, 0.1), (250, 0.1)],
    distributed_backend="mpi",
    rank=0,
    optimizer="all-reduce",
    n_workers=1,
    distributed_init_file=None,
    distributed_lr_warmup_epoch=5,
    distributed_lr_warmup_factor=1,
    use_decentralized_data=False,
    log_verbosity=1,
    seed=1,
    evaluate_average_model=True,
    spectrum_logging_params=[],
    spectrum_logging_epochs=[],
    spectrum_logging_worker_pairs=[],
)

output_dir = "./output.tmp"  # will be overwritten by run.py


def main():
    torch.manual_seed(config["seed"] + config["rank"])
    np.random.seed(config["seed"] + config["rank"])

    # Run on the GPU is one is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)

    timer = Timer(verbosity_level=config["log_verbosity"], log_fn=log_metric)

    init_distributed_pytorch()
    assert config["n_workers"] == torch.distributed.get_world_size()

    if torch.distributed.get_rank() == 0:
        if config["task"] == "Cifar":
            download_cifar()
        elif config["task"] == "LSTM":
            download_wikitext2()
    torch.distributed.barrier()

    task = tasks.build(task_name=config["task"], device=device, timer=timer, **config)

    local_optimizer = torch.optim.SGD(
        [
            {
                "params": [
                    p
                    for p, name in zip(task.state, task.parameter_names)
                    if parameter_type(name) == "batch_norm"
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for p, name in zip(task.state, task.parameter_names)
                    if parameter_type(name) != "batch_norm"
                ]
            },
        ],
        lr=config["learning_rate"],  # to correct for summed up gradients
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
        nesterov=(config["momentum"] > 0),
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(local_optimizer, learning_rate_schedule)

    topology = get_topology()
    optimizer = get_optimizer(timer, topology, task.state, local_optimizer.step)

    if "LSTM" in config["task"]:
        train_loader = task.train_iterator(config["batch_size"])
        batches_per_epoch = torch.tensor(len(train_loader))
        torch.distributed.all_reduce(batches_per_epoch, op=torch.distributed.ReduceOp.MIN)
        batches_per_epoch = batches_per_epoch.item()

    for epoch in range(config["num_epochs"]):
        timer.epoch = epoch

        epoch_metrics = MeanAccumulator()

        if not "LSTM" in config["task"]:
            train_loader = task.train_iterator(config["batch_size"])
            batches_per_epoch = len(train_loader)

        with timer("epoch.body"):
            my_rank = torch.distributed.get_rank()
            print(f"Worker {my_rank} starting epoch {epoch} with {len(train_loader)} batches")
            for i, batch in enumerate(train_loader):
                if i >= batches_per_epoch:
                    break
                epoch_frac = epoch + i / batches_per_epoch
                scheduler.step(
                    epoch + (i + 1) / batches_per_epoch
                )  # for compatibility with Choco code
                timer.epoch = epoch_frac
                info({"state.progress": epoch_frac / config["num_epochs"]})

                metrics = optimizer.step(lambda: task.batch_loss_and_gradient(batch))
                epoch_metrics.add(metrics)

        with timer("epoch.post"):
            for key, value in epoch_metrics.value().items():
                log_metric(
                    key,
                    {
                        "value": value.item(),
                        "epoch": epoch + 1.0,
                        "bits": optimizer.bits_sent,
                        "messages": optimizer.messages_sent,
                    },
                    tags={"split": "train"},
                )

        if (epoch + 1) in config["spectrum_logging_epochs"]:
            with timer("spectrum_logging"):
                print("spectrum logging at epoch {}".format(epoch + 1))
                my_rank = torch.distributed.get_rank()
                for working_node, sending_node in config["spectrum_logging_worker_pairs"]:
                    for param, name in zip(task.state, task.parameter_names):
                        # print(name)
                        if name in config["spectrum_logging_params"]:
                            if my_rank == sending_node:
                                print(f"{my_rank} sending {name}")
                                torch.cuda.synchronize()
                                torch.distributed.send(param, working_node)
                            elif my_rank == working_node:
                                print(f"{my_rank} receiving {name}")
                                other_workers_param = torch.empty_like(param)
                                torch.cuda.synchronize()
                                torch.distributed.recv(other_workers_param, sending_node)
                                u, s, v = torch.svd(
                                    (param - other_workers_param).view(param.shape[0], -1).cpu()
                                )
                                for i, val in enumerate(s):
                                    print(f"{i} / {val.cpu().item()}")
                                    log_metric(
                                        "spectrum",
                                        {"value": val.cpu().item(), "index": i},
                                        tags={
                                            "workers": f"{working_node}-{sending_node}",
                                            "parameter": name,
                                            "epoch": epoch + 1,
                                        },
                                    )
                                del u, s, v

        with timer("epoch.test"):
            test_stats = task.test()
            for key, value in test_stats.items():
                log_metric(
                    key,
                    {
                        "value": value.item(),
                        "epoch": epoch + 1.0,
                        "bits": optimizer.bits_sent,
                        "messages": optimizer.messages_sent,
                    },
                    tags={"split": "test"},
                )

            # Compute and test the average model + consensus distance
            buffer, shapes = pack([t.float() for t in task.state_dict().values()])
            local_buffer = buffer.clone()
            torch.distributed.all_reduce(buffer)
            buffer /= torch.distributed.get_world_size()
            if torch.distributed.get_rank() == 0:
                log_metric(
                    "consensus_distance",
                    {"value": (local_buffer - buffer).norm().item(), "epoch": epoch + 1.0},
                    {"type": "full_state_vector"},
                )
                if config["evaluate_average_model"]:
                    avg_model = {
                        key: value
                        for key, value in zip(task.state_dict().keys(), unpack(buffer, shapes))
                    }
                    test_stats = task.test(state_dict=avg_model)
                    for key, value in test_stats.items():
                        log_metric(
                            key,
                            {
                                "value": value.item(),
                                "epoch": epoch + 1.0,
                                "bits": optimizer.bits_sent,
                                "messages": optimizer.messages_sent,
                            },
                            tags={"split": "test_avg"},
                        )
            del local_buffer, buffer, shapes

        params_flat, shapes = pack(task.state)
        avg_params_flat = params_flat.clone()
        torch.distributed.all_reduce(avg_params_flat)
        avg_params_flat /= torch.distributed.get_world_size()
        if torch.distributed.get_rank() == 0:
            log_metric(
                "consensus_distance",
                {"value": (params_flat - avg_params_flat).norm().item(), "epoch": epoch + 1.0},
                {"type": "params_only"},
            )
        del params_flat, shapes, avg_params_flat

        for entry in timer.transcript():
            log_runtime(entry["event"], entry["mean"], entry["std"], entry["instances"])

    info({"state.progress": 1.0})


def get_topology():
    num_workers = torch.distributed.get_world_size()
    if config["topology"] == "ring":
        return topologies.Ring(num_workers)
    elif config["topology"] == "fully-connected":
        return topologies.FullyConnected(num_workers)
    elif config["topology"] == "torus":
        return topologies.Torus(num_workers)
    elif config["topology"] == "social-network":
        return topologies.SocialNetwork(num_workers)
    else:
        raise ValueError("Unknown topology {}".format(config["topology"]))


def get_optimizer(timer, topology, params, update_fn):
    if config["optimizer"] == "dpsgd":
        return algorithms.DPSGD(
            timer,
            algorithms.SimpleGossip(topology, diffusion_rate=config["optimizer_diffusion_rate"]),
            params,
            update_fn,
        )
    elif config["optimizer"] == "all-reduce":
        return algorithms.DPSGD(
            timer, algorithms.AllReduce(topology), params, update_fn, overlapping=False
        )
    elif config["optimizer"] == "power-gossip":
        return algorithms.DPSGD(
            timer,
            algorithms.OnlyOnLargeParameters(
                topology,
                algorithms.PowerGossip(
                    topology,
                    rank=config["optimizer_rank"],
                    num_iterations=config["optimizer_num_iterations"],
                    warm_start=config["optimizer_warm_start"],
                    diffusion_rate=config["optimizer_diffusion_rate"],
                    round_weights=config["optimizer_round_weights"],
                ),
            ),
            params,
            update_fn,
        )
    elif config["optimizer"] == "choco" or config["optimizer"] == "deepsqueeze":
        classType = {"choco": algorithms.ChocoGossip, "deepsqueeze": algorithms.DeepSqueezeGossip}[
            config["optimizer"]
        ]

        if config["optimizer_compressor"] == "top-k":
            compressor = compressors.TopK(keep_ratio=config["optimizer_keep_ratio"])
        elif config["optimizer_compressor"] == "svd":
            compressor = compressors.SVD(rank=config["optimizer_rank"])
        elif config["optimizer_compressor"] == "sign-and-norm":
            compressor = compressors.SignAndNorm()
        else:
            raise ValueError("Unknown compressor {}".format(config["optimizer_compressor"]))

        return algorithms.DPSGD(
            timer,
            classType(
                topology, diffusion_rate=config["optimizer_diffusion_rate"], compressor=compressor
            ),
            params,
            update_fn,
        )
    elif config["optimizer"] == "moniqua":
        return algorithms.DPSGD(
            timer,
            algorithms.MoniquaGossip(
                topology,
                diffusion_rate=config["optimizer_diffusion_rate"],
                theta=config["optimizer_theta"],
            ),
            params,
            update_fn,
        )
    else:
        raise ValueError("Unknown optimizer {}".format(config["optimizer"]))


def init_distributed_pytorch():
    if config["distributed_backend"] == "mpi":
        torch.distributed.init_process_group("mpi")
    else:
        if config["distributed_init_file"] is None:
            config["distributed_init_file"] = os.path.join(output_dir, "dist_init")
        torch.distributed.init_process_group(
            backend=config["distributed_backend"],
            init_method="file://" + os.path.abspath(config["distributed_init_file"]),
            timeout=datetime.timedelta(0, 1800),
            world_size=config["n_workers"],
            rank=config["rank"],
        )


def parameter_type(parameter_name):
    if "conv" in parameter_name and "weight" in parameter_name:
        return "convolution"
    elif re.match(r""".*\.bn\d+\.(weight|bias)""", parameter_name):
        return "batch_norm"
    else:
        return "other"


def learning_rate_schedule(epoch):
    """Apply any learning rate schedule"""
    lr = 1.0

    if config["distributed_lr_warmup_epoch"] > 0:
        warmup_epochs = config["distributed_lr_warmup_epoch"]
        max_factor = config["distributed_lr_warmup_factor"]
        factor = 1.0 + (max_factor - 1.0) * min(epoch / warmup_epochs, 1.0)
        lr *= factor

    for (milestone, factor) in config["lr_schedule_milestones"]:
        if epoch >= milestone:
            lr *= factor
        else:
            return lr
    return lr


def log_info(info_dict):
    """Add any information to MongoDB
       This function will be overwritten when called through run.py"""
    pass


def log_metric(name, values, tags={}):
    """Log timeseries data
       This function will be overwritten when called through run.py"""
    value_list = []
    for key in sorted(values.keys()):
        value = values[key]
        value_list.append(f"{key}:{value:7.3f}")
    values = ", ".join(value_list)
    tag_list = []
    for key, tag in tags.items():
        tag_list.append(f"{key}:{tag}")
    tags = ", ".join(tag_list)
    print("{name:30s} - {values} ({tags})".format(name=name, values=values, tags=tags))


def log_runtime(label, mean_time, std, instances):
    """This function will be overwritten when called through run.py"""
    pass


def info(*args, **kwargs):
    if config["rank"] == 0:
        log_info(*args, **kwargs)


def download_cifar(data_root=os.path.join(os.getenv("DATA"), "data")):
    import torchvision

    dataset = torchvision.datasets.CIFAR10
    training_set = dataset(root=data_root, train=True, download=True)
    test_set = dataset(root=data_root, train=False, download=True)


def download_wikitext2(data_root=os.path.join(os.getenv("DATA"), "data")):
    import torchtext

    torchtext.datasets.WikiText2.splits(
        torchtext.data.Field(lower=True), root=os.path.join(data_root, "wikitext2")
    )


if __name__ == "__main__":
    main()
