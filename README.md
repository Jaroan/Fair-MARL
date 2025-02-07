# Cooperation and Fairness in Multi-Agent Reinforcement Learning (Fair-MARL)

This repository contains the code for the paper **"[Cooperation and Fairness in Multi-Agent Reinforcement Learning](https://dl.acm.org/doi/full/10.1145/3702012)"**, which introduces a method to incorporate fairness for multi-agent navigation tasks. The method builds on the InforMARL framework and extends it to ensure fair cooperation in scenarios like MPE's simple spread (coverage) and formation.

Paper Link: :page_facing_up: :paperclip: [https://arxiv.org/abs/2410.14916](https://arxiv.org/abs/2410.14916)

Paper Website: :computer: :globe_with_meridians: [Website](https://jaroan.github.io/jasminejerrya/Fair_MARL.html)

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Environment](#environment)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Code Structure](#code-structure)
- [Results](#results)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Introduction

The Fair-MARL method addresses fairness in cooperative multi-agent reinforcement learning (MARL), where agents must not only achieve task success but also do so in a manner that promotes fairness in navigation for all agents. This is particularly relevant in tasks involving navigation, such as:

- **Coverage Navigation**: Agents must spread out to cover target locations.
- **Formation**: Agents must arrange themselves in specific formations.

Our approach extends the **InforMARL** framework to include fairness in the goal assignment and rewards, enabling agents to learn policies that are both efficient and fair. 

## Features

- **Fair Goal Assignment**: Incorporates fairness principles in the goal assignment process.
- **Fairness Reward**: Includes a fairness reward that is based on agents's distance traveled.
  
## Environment

The code is implemented for use with the **Multi-Agent Particle Environment (MPE)**, specifically for tasks like `simple_spread`. The environment simulates continuous spaces where agents must collaborate to achieve a common goal.

You can find the MPE environment here: [Multi-Agent Particle Environment (MPE)](https://github.com/openai/multiagent-particle-envs)

## Installation

To get started with the Fair-MARL method, clone this repository and install the required dependencies. Ensure you have pip version pip==23.1.2. Installing torch beforehand ensures the correct installation of other components.

> NOTE: Using a conda environment is preferred. Please use the following command to create a conda environment with the correct python version.

```
conda create -n fairmarl python=3.11
conda activate fairmarl
```

```bash
git clone https://github.com/yourusername/fair-marl.git
cd fair-marl
pip install pip==23.1.2
pip install torch==2.0.1
pip install -r requirements.txt
```



### Dependencies
- Python 3.11+
- PyTorch
- OpenAI Gym
- Multi-Agent Particle Environment (MPE)

## Usage

### Training

Training scripts are located in the folder ```train_scripts```. To train the Fair-MARL agents on the coverage tasks the command is alongg the following lines:

```bash
python -u onpolicy/scripts/train_mpe.py \
--project_name "test" \
--env_name "GraphMPE" \
--algorithm_name "rmappo" \
--seed 2 \
--experiment_name "test123" \
--scenario_name "navigation_graph"
```

This will train agents using the Fair-MARL method on the chosen task (`navigation_graph` in this case). Additional parameters for training, such as the number of agents, can be modified in the configuration file or passed as command-line arguments.

> NOTE 1: Please note that for training we have enabled wandb logging by default. Please inspect your logging mechanishm or use the flag `--use_wandb` to prevent wandb longging.

> NOTE 2a: Training any of the fairness aware models will require access to Gurobi optimization software. Please install it in your environment.

### Evaluation

After training, you can evaluate the trained agents by running:

```bash
python onpolicy/scripts/eval_mpe.py \
--model_dir='model_weights/FA_FR/unicycle_model' \
--render_episodes=2 \
--world_size=3 \
--num_agents=3 \
--num_obstacles=0 \
--seed=0 \
--num_landmarks=3 \
--episode_length=50 \
--use_dones=False \
--collaborative=False --model_name='FA' \
--scenario_name='nav_fairassign_fairrew_formation_graph' \
--dynamics_type='unicycle_vehicle' \
--goal_rew=30 \
--fair_rew=1 \
--save_gifs \
--use_render \
--num_walls=0 \
--zeroshift=5 \
--min_obs_dist 0.5 \
--total_actions 5
```

This will load the trained model and evaluate its performance in the specified environment. Additional parameters for evaluation, such as the number of agents, can be modified in the configuration file or passed as command-line arguments.

> NOTE 2b: Training any of the fairness aware models will require access to Gurobi optimization software. Please install it in your environment.


## Code Structure

```bash
.
├── README.md                     # Project Overview
├── license                       # Project license file
├── requirements.txt              # Dependencies
├── train_scripts                 # Training Script
├── eval_scripts                  # Sample Evaluation Script using Trained models
├── model_weights/                # Directory for saving trained models
├── utils/                        # Configuration files for different environments and algorithms
├── multi-agent/                    # Fair-MARL specific code
│   ├── custom-scenarios              # Core Fair-MARL Algorithm
│   ├── navigation_environment.py        # Fairness-based goal assignment logic
│   ├── agent.py                  # Multi-agent definitions
│   └── utils.py                  # Utility functions
└── onpolicy/                          # MPE environment files (if necessary)
```

- **`multiagent/custom_scenarios/navigation_graph.py`**: Implements the Fair-MARL reinforcement learning algorithm.
- **`marl_fair_assign.py`**: Contains the logic for fair goal assignment.

- We have created adocument detailing the network architecture here:
- We have created a document for easy understandng of our codebase here

## Results

Here we summarize the results from the experiments. The Fair-MARL method achieves **fairer goal assignment** and **better cooperation** compared to baseline methods. For example:

- **Coverage navigation**: Fair-MARL agents spread out more equitably to different target locations.
- **Formation**: Agents arrange themselves in stable formations while ensuring fairness in positional assignments.

For detailed results and analysis, please refer to our paper.

## Citation

If you find this repository helpful in your research, please cite the corresponding paper:

```bibtex
@article{aloor2024cooperation,
  title={Cooperation and Fairness in Multi-Agent Reinforcement Learning},
  author={Aloor, Jasmine and Nayak, Siddharth Nagar and Dolan, Sydney and Balakrishnan, Hamsa},
  journal={Journal on Autonomous Transportation Systems},
  year={2024},
  publisher={ACM New York, NY},
  doi={10.1145/3702012}
}

```

## Troubleshooting

1. Known issues with pytorch geometric and torch-scatter packege installation. Please refer to the requirements.txt to note the versions being used in the code.
 Correct order of installation of Pytorch Geometric packages if encountering any errors:
 ```bash
    pip install --verbose git+https://github.com/pyg-team/pyg-lib.git
    pip install --verbose torch_scatter
    pip install --verbose torch_sparse
    pip install --verbose torch_cluster
    pip install --verbose torch_spline_conv
```
2. Rendering issues with Linux users: Follow the instructions to access display for the visualization of evaluation tests.

## Questions/Requests

Please file an issue if you have any questions or requests about the code or the paper. If you prefer your question to be private, you can alternatively email me at jjaloor@mit.edu

## Related papers

1. InforMARL: [https://nsidn98.github.io/InforMARL/](https://nsidn98.github.io/InforMARL/) Paper: [Scalable Multi-Agent Reinforcement Learning through Intelligent Information Aggregation](https://arxiv.org/abs/2211.02127)
2. [Satellite Navigation and Coordination with Limited Information Sharing](https://arxiv.org/abs/2211.03658)

## Contributing

We would be happy to accept PRs that help extend or improve this work.