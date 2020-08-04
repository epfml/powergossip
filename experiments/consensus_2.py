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
script = "gossip_run.py"
description = """
Now using optimal scaling for the uncompressed baseline and PowerGossip.
""".strip()


code_package, files_uploaded = upload_code_package(
    ".", excludes=excluded_files + ["tasks", "train.py"]
)
print("Uploaded {} files.".format(len(files_uploaded)))


def schedule(name, config):
    # Skip pre-existing entries
    if mongo.job.count_documents({"project": project, "job": name, "experiment": experiment}) > 0:
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
    print(f'sbatch --ntasks {n_workers} --job-name="{name}" --wrap="srun jobrun {job_id} --mpi"')


for algorithm in [
    "uncompressed",
    "power_gossip",
    "power_gossip_random",
    "power_gossip_signs",
    "power_gossip_random_signs",
    "sign_choco",
    "topk_choco",
    "svd_choco",
    "moniqua",
]:
    for n_workers, topology in [(8, "ring")]:
        for data_type in ("random", "faces"):
            name = f"{algorithm}_{n_workers}{topology}"
            if data_type != "random":
                name += f"_{data_type}"
            config = {
                "n_workers": n_workers,
                "topology": topology,
                "algorithm": algorithm,
                "data_type": data_type,
                "hyperparameter_range_max": 1.308,  # optimal for 8-ring
                "num_steps": 5000,
            }
            if data_type == "faces":
                config["num_steps"] = 1000
            if algorithm == "uncompressed":
                config["num_steps"] = config["num_steps"] // 10
            if algorithm == "uncompressed" or "power" in algorithm:
                config["hyperparameter_range_min"] = config["hyperparameter_range_max"]

            if data_type == "faces" and "power" in algorithm:
                config["num_steps"] = config["num_steps"] * 3
            if data_type == "faces" and "topk" in algorithm:
                config["num_steps"] = config["num_steps"] * 2

            if not "choco" in algorithm:
                config["hyperparameter_num_samples"] = 1
            if "moniqua" in algorithm:
                config["theta_num_samples"] = 10
                config["hyperparameter_num_samples"] = 10
            else:
                config["theta_num_samples"] = 1
            schedule(name, config)
