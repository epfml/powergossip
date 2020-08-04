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


seed = 0
for factor in [4]:
    for theta in [0.25]:
        for diffusion_rate in [5e-3]:
            schedule(
                f"moniqua-lr{factor}-theta{theta}-dr{diffusion_rate}",
                dict(
                    distributed_lr_warmup_factor=factor,
                    optimizer="moniqua",
                    optimizer_diffusion_rate=diffusion_rate,
                    optimizer_theta=theta,
                    seed=seed,
                ),
                skip_existing=True,
            )
