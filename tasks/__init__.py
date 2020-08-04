def build(task_name, seed, device, timer, **kwargs):
    if task_name == "Cifar":
        from .cifar import CifarTask

        return CifarTask(
            seed=seed,
            device=device,
            timer=timer,
            architecture=kwargs.get("task_architecture", "ResNet18"),
            use_decentralized_data=kwargs["use_decentralized_data"],
            decentralized_data_shuffled_ratio=kwargs.get("decentralized_data_shuffled_ratio", 0.0),
        )

    elif task_name == "LSTM":
        from .lstm import LanguageModelingTask

        return LanguageModelingTask(
            seed=seed, device=device, timer=timer, batch_size=kwargs["batch_size"]
        )

    else:
        raise ValueError("Unknown task name")
