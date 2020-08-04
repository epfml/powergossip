import numpy as np
import torch

import compressors
import algorithms
import topologies


config = {
    "seed": 42,
    "algorithm": "choco",
    "topology": "ring",
    "matrix_size": 100,
    "hyperparameter_num_samples": 20,
    "hyperparameter_range_min": 1e-4,
    "hyperparameter_range_max": 1.308,
    "theta_num_samples": 20,
    "theta_range_min": 0.1,
    "theta_range_max": 10,
    "num_steps": 5000,
    "data_type": "random",
    "compression_rank": 1,
    "compression_num_iterations": 1,
}


def main():
    torch.distributed.init_process_group("mpi")

    reset_randomness()

    world = get_topology()

    problem = GossipProblem(
        (config["matrix_size"], config["matrix_size"]), data_type=config["data_type"]
    )

    hyperparameters_to_try = np.exp(
        np.linspace(
            np.log(config["hyperparameter_range_min"]),
            np.log(config["hyperparameter_range_max"]),
            config["hyperparameter_num_samples"],
        )
    )

    thetas_to_try = np.exp(
        np.linspace(
            np.log(config["theta_range_min"]),
            np.log(config["theta_range_max"]),
            config["theta_num_samples"],
        )
    )

    log_info({"state.progress": 0.0})
    for hpo_nr, diffusion_parameter in enumerate(hyperparameters_to_try):
        for theta in thetas_to_try:
            params = [problem.init()]
            algorithm = get_algorithm(world, diffusion_parameter, theta)
            state = algorithm.init_state(params)
            tags = {}
            if not "power" in config["algorithm"] and not "uncompressed" in config["algorithm"]:
                tags["diffusion_rate"] = str(diffusion_parameter.item())
            if "moniqua" in config["algorithm"]:
                tags["theta"] = str(theta.item())

            error = problem.error(params[0])
            initial_error = error.clone()
            if torch.distributed.get_rank() == 0:
                log_metric(
                    "error",
                    {
                        "value": error.item(),
                        "epoch": 0,
                        "bits": state.bits_sent,
                        "messages": state.messages_sent,
                    },
                    tags,
                )

            for step in range(config["num_steps"]):
                params, state = algorithm.step(params, state)
                steps_taken = step + 1
                if (
                    steps_taken < 10
                    or (steps_taken < 100 and steps_taken % 10 == 0)
                    or (steps_taken % 100) == 0
                ):
                    error = problem.error(params[0])
                    if torch.distributed.get_rank() == 0:
                        log_metric(
                            "error",
                            {
                                "value": error.item(),
                                "epoch": steps_taken,
                                "bits": state.bits_sent,
                                "messages": state.messages_sent,
                            },
                            tags,
                        )
                    if torch.isnan(error).any() or error.item() > initial_error.item():
                        continue
        log_info({"state.progress": (hpo_nr + 1) / len(hyperparameters_to_try)})


def reset_randomness():
    seed = config["seed"] + torch.distributed.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_topology():
    if config["topology"] == "ring":
        return topologies.Ring(torch.distributed.get_world_size())
    if config["topology"] == "fully-connected":
        return topologies.FullyConnected(torch.distributed.get_world_size())
    if config["topology"] == "torus":
        return topologies.Torus(torch.distributed.get_world_size())
    else:
        raise ValueError("Unknown topology {}".format(config["topology"]))


def get_algorithm(world, diffusion_parameter, theta):
    if config["algorithm"] == "svd_choco":
        return algorithms.ChocoGossip(
            world, diffusion_parameter, compressors.SVD(rank=config["compression_rank"])
        )
    elif config["algorithm"] == "topk_choco":
        return algorithms.ChocoGossip(
            world, diffusion_parameter, compressors.TopK(rank=config["compression_rank"])
        )
    elif config["algorithm"] == "sign_choco":
        return algorithms.ChocoGossip(world, diffusion_parameter, compressors.SignAndNorm())
    elif config["algorithm"] == "moniqua":
        return algorithms.MoniquaGossip(world, diffusion_parameter, theta)
    elif config["algorithm"] == "fp_choco":
        return algorithms.ChocoGossip(world, diffusion_parameter, compressors.Identity())
    elif config["algorithm"] == "power_gossip":
        return algorithms.PowerGossip(
            world,
            rank=config["compression_rank"],
            num_iterations=config["compression_num_iterations"],
            warm_start=True,
            diffusion_rate=diffusion_parameter,
        )
    elif config["algorithm"] == "power_gossip_random":
        return algorithms.PowerGossip(
            world,
            rank=config["compression_rank"],
            num_iterations=config["compression_num_iterations"],
            warm_start=False,
            diffusion_rate=diffusion_parameter,
        )
    elif config["algorithm"] == "power_gossip_random_signs":
        return algorithms.PowerGossip(
            world,
            rank=config["compression_rank"],
            num_iterations=config["compression_num_iterations"],
            warm_start=False,
            diffusion_rate=diffusion_parameter,
            round_weights=True,
        )
    elif config["algorithm"] == "power_gossip_signs":
        return algorithms.PowerGossip(
            world,
            rank=config["compression_rank"],
            num_iterations=config["compression_num_iterations"],
            warm_start=True,
            diffusion_rate=diffusion_parameter,
            round_weights=True,
        )
    elif config["algorithm"] == "uncompressed":
        return algorithms.SimpleGossip(world, diffusion_rate=diffusion_parameter)
    else:
        raise ValueError("Unknown algorithm {}".format(config["algorithm"]))


class GossipProblem:
    def __init__(self, matrix_shape, data_type):
        if data_type == "random":
            self.starting_state = torch.randn(matrix_shape)
        elif data_type == "offset":
            self.starting_state = torch.randn(matrix_shape) + torch.distributed.get_rank()
        elif data_type == "sparse":
            self.starting_state = torch.zeros(matrix_shape, dtype=torch.float32)
            self.starting_state.view(-1)[torch.distributed.get_rank()] += 1.0
        elif data_type == "log-normal":
            self.starting_state = torch.exp(torch.randn(matrix_shape))
        elif data_type == "faces":
            import sklearn.datasets

            dataset = sklearn.datasets.fetch_olivetti_faces()
            images = dataset["images"]
            np.random.RandomState(0).shuffle(images)
            self.starting_state = torch.from_numpy(images[torch.distributed.get_rank()])
        else:
            raise ValueError("Unknown data type {}".format(data_type))

        # Compute the mean of the starting state across workers
        self.target = self.starting_state.clone()
        torch.distributed.all_reduce(self.target)
        self.target /= torch.distributed.get_world_size()

    def init(self):
        """Generates starting parameters"""
        return self.starting_state.clone()

    def error(self, parameters):
        """Avg Error/loss"""
        error = torch.sum((parameters - self.target) ** 2)
        torch.distributed.all_reduce(error)
        error /= torch.distributed.get_world_size()
        return error


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


def log_info(info_dict):
    """Add any information to MongoDB
       This function will be overwritten when called through run.py"""
    pass


if __name__ == "__main__":
    main()
