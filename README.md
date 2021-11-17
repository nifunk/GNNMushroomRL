# GNNMushroomRL

This repo contains the implementation of the graph neural networks as well
as the learning algorithms presented in the work 
[Learn2Assemble with Structured Representations and Search for Robotic Architectural Construction](https://sites.google.com/view/learn2assemble) by N. Funk, G. Chalvatzaki, B. Belousov and J. Peters, which has been accepted in the Conference on Robot Learning (CoRL) 2021.

Additional video material can accesssed [here](https://sites.google.com/view/learn2assemble).

If you use code or ideas from this work for your projects or research, please cite it.

```
@inproceedings{
funk2021learnassemble,
title={Learn2Assemble with Structured Representations and Search for Robotic Architectural Construction},
author={Niklas Funk and Georgia Chalvatzaki and Boris Belousov and Jan Peters},
booktitle={5th Annual Conference on Robot Learning },
year={2021},
url={https://openreview.net/forum?id=wBT0lZJAJ0V}
}
```

This repository is heavily based on the MushroomRL library and contains multiple important modifications
such as the graph based representations, parallelized rollouts, and
some advanced logging functionality.

## Installation

In case you want to use the repository as presented in the Learn2Assemble paper, it is required to install the 
[Learn2Assemble Repository](https://github.com/nifunk/learn2assemble) which contains all of the 3D assembly environments.
Please refer to this repository for the related installation instructions.

In addition to the instructions in [there](https://github.com/nifunk/learn2assemble), this repository further requires 
additional dependencies which can be installed by executing the shell script (after activating the environment)
'install_additionals.sh', which is placed on the top level of this repository, i.e. execute:

```
./install_additionals.sh
```

Finally perform a local installation of the package via:

```
pip install -e .
```

## Usage

In the folder named trained_models, we provide the trained models which were the result of running our training procedure,
as well as the exemplary training commands how to obtain them.
Please see the README inside this folder for further instructions.

**Note: this branch is specifically aimed to load our pretrained models (which have been trained using cuda) on non-cuda
machines. This has been achieved by adapting the loading procedure.**

**We recommend to only use this branch for the aforementioned purpose. If you want to use our code as the basis to 
develop something new / train models from scratch, please do use the main branch**.

# Credits

This repository is based on previous work.

* Most importantly the work heavily relies on the [MushroomRL library](https://github.com/MushroomRL/mushroom-rl)

* We also used parts from [SimuRLacra](https://github.com/famura/SimuRLacra), especially to allow the parallelization
of the training

* For the implementations of the S2V and MHA graph neural networks, we have
used Code from the following two repositories: [Attention, Learn to Solve Routing Problems!](https://github.com/wouterkool/attention-learn-to-route) 
and [Exploratory Combinatorial Optimization with Reinforcement Learning](https://github.com/tomdbar/eco-dqn).
