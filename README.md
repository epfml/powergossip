# PowerGossip: Practical Low-Rank Communication Compression in Decentralized Deep Learning

Abstract:

Lossy gradient compression has become a practical tool to overcome the communication bottleneck in centrally coordinated distributed training of machine learning models.
However, algorithms for decentralized training with compressed communication over arbitrary connected networks have been more complicated, requiring additional memory and hyperparameters.
We introduce a simple algorithm that directly compresses the model differences between neighboring workers using low-rank linear compressors applied on model differences.
Inspired by the PowerSGD algorithm for centralized deep learning, this algorithm uses power iteration steps to maximize the information transferred per bit.
We prove that our method requires no additional hyperparameters, converges faster than prior methods, and is asymptotically independent of both the network and the compression.
Out of the box, these compressors perform on par with state-of-the-art tuned compression algorithms in a series of deep learning benchmarks.

# Code

-   [train.py](train.py) is the entrypoint for deep learning experiments.
-   [gossip_run.py](gossip_run.py) is the entrypoint for consensus experiments.
-   [algorithms.py](algorithms.py) implements decentralized consensus and learning algorithms.
-   [compressors.py](compressors.py) implements compressors for [ChocoSGD](https://github.com/epfml/ChocoSGD) and `DeepSqueeze`.
-   Experiment scheduling code for our experiments is listed under [experiments](experiments/).

# Running and configuring experiments

You can override the global variables `config`, `log_metric` and `output_dir` from [train.py](train.py) before running `train.main()`:

```python
import train

# Configure the worker
train.config["n_workers"] = 4
train.config["rank"] = 0

train.output_dir = "whatever you like"
train.log_info = your_function_pointer
train.log_metric = your_metric_function_pointer

train.main()
```

# Distributed training

We use a separate process per “worker”.
We use MPI (`mpirun`) to create these processes.

# Environment

We provide a `setup.sh` file under [environment](environment)
that describes our computation environment.
