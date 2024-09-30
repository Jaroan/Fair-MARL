# Cooperation and Fairness in Multi-Agent Reinforcement Learning (FAIR-MARL)

This repository contains the code for the paper **"Cooperation and Fairness in Multi-Agent Reinforcement Learning"**, which introduces a method to incorporate fairness for multi-agent navigation tasks. The method builds on the InforMARL framework and extends it to ensure fair cooperation in scenarios like MPE's simple spread (coverage) and formation.

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

The FAIR-MARL method addresses fairness in cooperative multi-agent reinforcement learning (MARL), where agents must not only achieve task success but also do so in a manner that promotes fairness in navigation for all agents. This is particularly relevant in tasks involving navigation, such as:

- **Simple Spread**: Agents must spread out to cover target locations.
- **Formation**: Agents must arrange themselves in specific formations.

Our approach extends the **InforMARL** framework to include fairness in the goal assignment and rewards, enabling agents to learn policies that are both efficient and fair. 

## Features

- **Fair Goal Assignment**: Incorporates fairness principles in the goal assignment process.
- **Fairness Reward**: Includes a fairness reward that is based on agents's distance traveled.
  
## Environment

The code is implemented for use with the **Multi-Agent Particle Environment (MPE)**, specifically for tasks like `simple_spread`. The environment simulates continuous spaces where agents must collaborate to achieve a common goal.

You can find the MPE environment here: [Multi-Agent Particle Environment (MPE)](https://github.com/openai/multiagent-particle-envs)

## Installation

To get started with the FAIR-MARL method, clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/fair-marl.git
cd fair-marl
pip install -r requirements.txt
```

### Dependencies
- Python 3.7+
- PyTorch
- OpenAI Gym
- Multi-Agent Particle Environment (MPE)

## Usage

### Training

To train the FAIR-MARL agents on the coverage tasks run the following command:

```bash
python train_mpe.py --env simple_spread --algorithm fair-marl
```

This will train agents using the FAIR-MARL method on the chosen task (`simple_spread` in this case). Additional parameters for training, such as the number of agents, can be modified in the configuration file or passed as command-line arguments.

### Evaluation

After training, you can evaluate the trained agents by running:

```bash
python eval_mpe.py --model_dir ./models/fair-marl/simple_spread
```

This will load the trained model and evaluate its performance in the specified environment.

## Code Structure

```bash
.
├── README.md                     # Project Overview
├── license                       # Project license file
├── requirements.txt              # Dependencies
├── train_scripts                 # Training Script
├── eval_scripts                  # Evaluation Script
├── model_weights/                # Directory for saving trained models
├── utils/                        # Configuration files for different environments and algorithms
├── multi-agent/                    # FAIR-MARL specific code
│   ├── custom-scenarios              # Core FAIR-MARL Algorithm
│   ├── navigation_environment.py        # Fairness-based goal assignment logic
│   ├── agent.py                  # Multi-agent definitions
│   └── utils.py                  # Utility functions
└── onpolicy/                          # MPE environment files (if necessary)
```

- **`fair_marl/algorithm.py`**: Implements the FAIR-MARL reinforcement learning algorithm.
- **`fairness_module.py`**: Contains the logic for fair goal assignment.
- **`agent.py`**: Defines the multi-agent RL structure and policy updates.
- **`train.py`**: Script to launch the training process.
- **`evaluate.py`**: Script to evaluate the performance of the trained agents.

## Results

Here we summarize the results from the experiments. The FAIR-MARL method achieves **fairer goal assignment** and **better cooperation** compared to baseline methods. For example:

- **Simple Spread**: FAIR-MARL agents spread out more equitably to different target locations.
- **Formation**: Agents arrange themselves in stable formations while ensuring fairness in positional assignments.

For detailed results and analysis, please refer to our paper.

## Citation

If you find this repository helpful in your research, please cite the corresponding paper:

```bibtex
@article{aloor2023fairmarl,
  title={Cooperation and Fairness in Multi-Agent Reinforcement Learning},
  author={ },
  journal={ACM Journal of Autonomous Transportation Systems},
  year={2023},
  volume={XX},
  pages={YY-ZZ},
}
```

## Acknowledgements

