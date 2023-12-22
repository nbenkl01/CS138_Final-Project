# CS 138: Reinforcement Learning--Final Project
## Exploring Deep Reinforcement Learning in Super Mario Bros.
### Noam Benkler
---
This repository accompanies the final project paper titled "How to Teach a Plumber? Reinforcement Learning in Super Mario Bros.". The project focuses on employing various reinforcement learning (RL) techniques, including Double Deep Q-Networks (DDQN) and Actor-Critic methods, within the Super Mario Bros. video game environment. The codebase provides implementations of different RL agents, environments, trainers, and evaluators to facilitate comprehensive experiments and evaluations.


# Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Usage](#usage)
- [Project Assumptions](#project-assumptions)
- [Additional Information](#additional-info)

# Overview <a name="overview"></a>

### Directory Structure
---

- **agents/**: Contains implementations of RL agents including Double Deep Q-Networks and Actor-Critic networks.
- **environments/**: Includes environments like actor-critic and Q-learning environments along with necessary wrappers.
- **trainers/**: Provides trainers for various algorithms such as A3C, DDQN, and PPO.
- **testers/**: Contains evaluation scripts for A3C, DDQN, and PPO.
- **utils/**: Houses utility functions, including metrics calculation.

### Files
---

**requirements.txt**: Lists the Python packages and dependencies required for running the code. Install these using pip install -r requirements.txt.

**test_main.py**: A script that orchestrates the testing and evaluation process for the trained RL agents. Executes the evaluation scripts for A3C, DDQN, and PPO.

**train_main.py**: A script that coordinates the training process for the RL agents. Initiates the training of DDQN, A3C, and PPO agents based on the specified configurations.

`agents/`
- **ACnetworks.py:** Defines the neural network architecture for the Actor and Critic components used in the Actor-Critic agents (A3C and PPO).

- **Qagent.py:** Implements the Q-learning agent using Double Deep Q-Networks (DDQN). Contains the Q-network architecture and training logic.

- **Qnetworks.py:** Provides the neural network architecture for both the baseline and augmented DDQN agents, including the Double Deep Q-Network and the Deep Recurrent Double Q-Network (DRDQN).

`environments/`
- **actor_critic_env.py:** Defines the environment for the Actor-Critic agents (A3C and PPO). Handles interaction with the Super Mario Bros. environment and observation processing.

- **qlearning_env.py:** Implements the environment for the Q-learning agent (DDQN). Manages interactions with the Super Mario Bros. environment and observation preprocessing.

- **wrappers.py:** Contains wrappers for preprocessing observations, such as grayscaling, resizing, and frame stacking, to prepare input for the neural networks.

`testers/`
- **a3c_eval.py:** Evaluates the trained A3C (Asynchronous Advantage Actor-Critic) agent on different Super Mario Bros. levels and provides performance metrics.

- **ddqn_eval.py:** Evaluates the trained DDQN (Double Deep Q-Network) agent on various levels and reports performance metrics.

- **ppo_eval.py:** Conducts evaluations on the PPO (Proximal Policy Optimization) agent, measuring its performance on different levels and generating relevant metrics.

``trainers/``
- **a3c_trainer.py:** Implements the training logic for the A3C (Asynchronous Advantage Actor-Critic) agent, including asynchronous interactions and global model updates.

- **ddqn_trainer.py:** Contains the training logic for the DDQN (Double Deep Q-Network) agent, incorporating experience replay and target network updates.

- **ppo_trainer.py:** Manages the training process for the PPO (Proximal Policy Optimization) agent, including policy updates, advantage computation, and global model adjustments.

`utils/`
- **metrics.py:** Provides utility functions for logging training metrics such as episode length, rewards, and losses.

# Setup  <a name="setup"></a>
### Requirements
--- 
Ensure you have the required packages installed by running:

```bash
pip install -r requirements.txt
```

# Usage  <a name="usage"></a>
### Training
---
To train the agents, execute the following:

``` bash
python train_main.py
```

This will initiate the training process for the DDQN, A3C, and PPO agents.

### Testing
---
To evaluate the trained agents, move the final checkpoints for each of the DDQN models from the `./checkpoints` directory which will have been created during training to a new, first-level, directory titled 'final_models', and run:

``` bash
python test_main.py
```

The evaluators (a3c_eval.py, ddqn_eval.py, ppo_eval.py) will provide an output dataframe with the performance metrics for each of the trained models.


# Project Assumptions <a name="project-assumptions"></a>

The project focuses on enhancing the performance of RL agents in the Super Mario Bros. environment by implementing algorithmic enhancements, including recurrent Long Short-Term Memory (LSTM) layers and distinct training curricula.
Two main RL techniques are explored: Double Deep Q-Networks (DDQN) and Actor-Critic methods (A3C and PPO).
The environment provided by OpenAI Gym's Super Mario Bros. serves as the testbed for training and evaluating the RL agents.

# Additional Information  <a name="additional-info"></a>

For a comprehensive understanding of the project, please refer to the associated paper by Noam Benkler titled "How to Teach a Plumber? Reinforcement Learning in Super Mario Bros." The paper delves deep into the theoretical aspects, experimental design, results, and implications of employing RL techniques within the Super Mario Bros. environment.
