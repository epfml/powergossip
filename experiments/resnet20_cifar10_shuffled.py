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
This is meant to be equivalent to the setup in the Choco DL paper
""".strip()
base_config = {
    "n_workers": 8,
    "topology": "ring",
    "batch_size": 128,
    "task_architecture": "ResNet20",
    "lr_schedule_milestones": [(150, 0.1), (225, 0.1)],
    "optimizer_diffusion_rate": 1.308,
}


code_package, files_uploaded = upload_code_package(".", excludes=excluded_files + ["gossip_run.py"])
print("Uploaded {} files.".format(len(files_uploaded)))


def schedule(name, config, skip_existing=False):
    # Skip pre-existing entries
    if (
        skip_existing
        and mongo.job.count_documents({"project": project, "job": name, "experiment": experiment})
        > 0
    ):
        return
    config = {**base_config, **config}
    n_workers = config["n_workers"]
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
    print(
        f'sbatch --ntasks {n_workers} --job-name="{name}" --gpus-per-task=1 --cpus-per-task=8 --wrap="srun jobrun {job_id} --mpi"'
    )


seed = 10

for optimizer in ["all-reduce", "dpsgd"]:
    for lr in [8, 11.3, 16]:
        schedule(
            f"{optimizer}-lr{lr}",
            dict(distributed_lr_warmup_factor=lr, optimizer=optimizer, seed=seed),
            skip_existing=True,
        )

for optimizer in ["all-reduce", "dpsgd"]:
    for lr in [11.3]:
        for seed in [20, 30, 40, 50, 60]:
            schedule(
                f"{optimizer}-lr{lr}-{seed}",
                dict(distributed_lr_warmup_factor=lr, optimizer=optimizer, seed=seed),
                skip_existing=True,
            )

for optimizer in ["choco"]:
    for lr in [16]:
        for seed in [10, 20, 30, 40, 50, 60]:
            schedule(
                f"{optimizer}-sign-lr{lr}-{seed}",
                dict(
                    distributed_lr_warmup_factor=lr,
                    optimizer_diffusion_rate=0.45,
                    optimizer_compressor="sign-and-norm",
                    optimizer=optimizer,
                    seed=seed,
                ),
                skip_existing=True,
            )

for optimizer in ["choco"]:
    for lr in [11.3]:
        for seed in [10, 20, 30, 40, 50, 60]:
            schedule(
                f"{optimizer}-sign-untuned-{seed}",
                dict(
                    distributed_lr_warmup_factor=lr,
                    optimizer_compressor="sign-and-norm",
                    optimizer=optimizer,
                    seed=seed,
                ),
                skip_existing=True,
            )


for optimizer in ["deepsqueeze"]:
    for lr in [4.8]:
        for seed in [10, 20, 30, 40, 50, 60]:
            schedule(
                f"{optimizer}-sign-lr{lr}-{seed}",
                dict(
                    distributed_lr_warmup_factor=lr,
                    optimizer_diffusion_rate=0.01,
                    optimizer_compressor="sign-and-norm",
                    optimizer=optimizer,
                    seed=seed,
                ),
                skip_existing=True,
            )


for optimizer in ["choco"]:
    for lr in [11.3]:
        for seed in [10, 20, 30, 40, 50, 60]:
            schedule(
                f"{optimizer}-topk-lr{lr}-{seed}",
                dict(
                    distributed_lr_warmup_factor=lr,
                    optimizer_diffusion_rate=0.0375,
                    optimizer_compressor="top-k",
                    optimizer_keep_ratio=0.01,
                    optimizer=optimizer,
                    seed=seed,
                ),
                skip_existing=True,
            )

for optimizer in ["power-gossip"]:
    for lr in [11.3]:
        for seed in [10, 20, 30, 40, 50, 60]:
            schedule(
                f"{optimizer}-lr{lr}-{seed}",
                dict(
                    distributed_lr_warmup_factor=lr,
                    optimizer_rank=1,
                    optimizer_num_iterations=1,
                    optimizer_warm_start=True,
                    optimizer_round_weights=False,
                    optimizer=optimizer,
                    seed=seed,
                ),
                skip_existing=True,
            )


for optimizer in ["power-gossip"]:
    for lr in [11.3]:
        for seed in [10, 20, 30, 40]:
            schedule(
                f"{optimizer}-rand-lr{lr}-{seed}",
                dict(
                    distributed_lr_warmup_factor=lr,
                    optimizer_rank=1,
                    optimizer_num_iterations=1,
                    optimizer_warm_start=False,
                    optimizer_round_weights=False,
                    optimizer=optimizer,
                    seed=seed,
                ),
                skip_existing=True,
            )

for optimizer in ["power-gossip"]:
    for lr in [11.3]:
        for seed in [10, 20, 30, 40, 50, 60]:
            schedule(
                f"{optimizer}-2it-lr{lr}-{seed}",
                dict(
                    distributed_lr_warmup_factor=lr,
                    optimizer_rank=1,
                    optimizer_num_iterations=2,
                    optimizer_warm_start=True,
                    optimizer_round_weights=False,
                    optimizer=optimizer,
                    seed=seed,
                ),
                skip_existing=True,
            )
for optimizer in ["power-gossip"]:
    for lr in [11.3]:
        for seed in [10, 20, 30, 40]:
            schedule(
                f"{optimizer}-2it-rand-lr{lr}-{seed}",
                dict(
                    distributed_lr_warmup_factor=lr,
                    optimizer_rank=1,
                    optimizer_num_iterations=2,
                    optimizer_warm_start=False,
                    optimizer_round_weights=False,
                    optimizer=optimizer,
                    seed=seed,
                ),
                skip_existing=True,
            )

# for optimizer in ["choco"]:
#     for lr in [16]:
#         for seed in [10, 20, 30, 40, 50]:
#             schedule(
#                 f"{optimizer}-topk-lr{lr}-{seed}",
#                 dict(
#                     distributed_lr_warmup_factor=lr,
#                     optimizer_diffusion_rate=0.0375,
#                     optimizer_compressor="top-k",
#                     optimizer_rank=1,
#                     optimizer=optimizer,
#                     seed=seed,
#                 ),
#                 skip_existing=True,
#             )

for seed in [10, 20, 30, 40, 50, 60]:
    schedule(
        f"moniqua-{seed}",
        dict(
            distributed_lr_warmup_factor=4,
            optimizer="moniqua",
            optimizer_diffusion_rate=5e-3,
            optimizer_theta=0.25,
            seed=seed,
        ),
        skip_existing=True,
    )
