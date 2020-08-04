#!/usr/bin/env python3

import os

from jobmonitor.api import (
    kubernetes_schedule_job,
    kubernetes_schedule_job_queue,
    register_job,
    upload_code_package,
)
from jobmonitor.connections import mongo

excluded_files = [
    "core",
    "output.tmp",
    ".vscode",
    "node_modules",
    "scripts",
    ".git",
    "*.pyc",
    "._*",
    "__pycache__",
    "*.pdf",
    "*.js",
    "*.yaml",
    ".pylintrc",
    ".gitignore",
    ".AppleDouble",
    ".jobignore",
]


project = "decentralized_powersgd"
experiment = os.path.splitext(os.path.basename(__file__))[0]
script = "train.py"
description = """
At some stages during training, this logs the singular values of parameter differences between a couple of worker pairs.
""".strip()
n_workers = 16
base_config = dict(
    task="LSTM",
    batch_size=64,  # per worker
    weight_decay=0,
    num_epochs=90,
    lr_schedule_milestones=[(60, 0.1), (80, 0.1)],
    learning_rate=1.25,
    momentum=0.0,
    optimizer_diffusion_rate=1,
    topology="ring",
    use_decentralized_data=True,
    n_workers=n_workers,
    evaluate_average_model=False,
    spectrum_logging_params=["encoder.weight", "rnn.weight_ih_l1"],
    spectrum_logging_epochs=[1, 50, 65, 89],
    spectrum_logging_worker_pairs=[(0, 1), (4, 5), (8, 9)],
)


code_package, files_uploaded = upload_code_package(".", excludes=excluded_files)
print("Uploaded {} files.".format(len(files_uploaded)))

ids = []


def schedule(name, config, skip_existing=True):
    # Skip pre-existing entries
    config = {**base_config, **config}
    if (
        skip_existing
        and mongo.job.count_documents(
            {
                "project": project,
                "job": name,
                "experiment": experiment,
                "config.learning_rate": config["learning_rate"],
            }
        )
        > 0
    ):
        return
    job_id = register_job(
        user="vogels",
        project=project,
        experiment=experiment,
        job=name,
        n_workers=n_workers,
        priority=10,
        config_overrides=config,
        runtime_environment={"clone": {"code_package": code_package}, "script": script},
        annotations={"description": description},
    )
    ids.append(job_id)


for seed in [10]:
    for optimizer in ["power-gossip"]:
        for lr in [38]:
            for its in [32]:
                schedule(
                    f"{optimizer}-{its}it-lr{lr}-{seed}",
                    dict(
                        distributed_lr_warmup_factor=lr,
                        optimizer_rank=1,
                        optimizer_num_iterations=its,
                        optimizer_warm_start=True,
                        optimizer_round_weights=False,
                        optimizer=optimizer,
                        seed=seed,
                    ),
                    skip_existing=True,
                )

ids_string = " ".join(ids)
print(
    """./cluster.py exec --num-workers 8 --machine-type n1-standard-8 --num-gpus-per-worker 1 --gpu-type nvidia-tesla-v100 --zone us-central1-a --cmd "/opt/anaconda3/bin/python -m spacy download en" """
)
print(
    f"""for id in {ids_string}; do sbatch -N 1 --ntasks-per-node 16 --cpus-per-task 2 --gres gpu:4 --gpu-bind=map_gpu:0,1,2,3 --job-name="$id" --wrap="srun jobrun $id --mpi"; done"""
)
