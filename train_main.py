import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import datetime
from pathlib import Path

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from utils.metrics import DDQNMetricLogger
from agents.Qagent import Mario

from environments.qlearning_env import SuperMarioGame

from trainers import ddqn_trainer, a3c_trainer, ppo_trainer

# Set environment variables for better compatibility

def baseline_main(total_episodes, render=False, recurrent=False):
    """
    Main function for training a baseline DDQN agent.

    Parameters:
    - total_episodes: Total number of training episodes
    - render: Whether to render the environment during training
    - recurrent: Whether to use a recurrent neural network
    """
    # Create the Super Mario game environment
    game = SuperMarioGame(world=1, stage=1, version=0, movement=SIMPLE_MOVEMENT)
    
    # Transform the environment for training
    env = game.transform(skip=4, gray=True, resize=84, transform=255, stack=4)
    
    env.reset()

    # Set up the save directory based on whether recurrent network is used
    if recurrent:
        save_dir = Path('checkpoints/lstm') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    else:
        save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    save_dir.mkdir(parents=True)

    checkpoint = None
    # Create the DDQN agent
    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint, recurrent=recurrent)

    # Create a logger for metrics
    logger = DDQNMetricLogger(save_dir)

    # Create a DDQN Trainer
    trainer = ddqn_trainer.Trainer(agent=mario, env=env, episodes=total_episodes, logger=logger)
    trainer.train(render=False)

def curriculum_experiment_main(total_episodes, date, basis='version', render=False):
    """
    Main function for conducting curriculum experiments.

    Parameters:
    - total_episodes: Total number of training episodes
    - date: Current date for creating save directories
    - basis: Basis for curriculum generation ('version' or 'movement')
    - render: Whether to render the environment during training
    """
    # Perform curriculum experiment to generate curricula
    experiments, teacher = ddqn_trainer.curriculum_experiment(total_episodes, curriculum_basis=basis)

    for experiment, curriculum in experiments.items():
        # Set up the save directory based on experiment and curriculum
        save_dir = Path(f'checkpoints/{date}/{basis}/{experiment}')
        save_dir.mkdir(parents=True)

        checkpoint = None
        # Create the DDQN agent
        agent = Mario(state_dim=(4, 84, 84), action_dim=teacher.games[0].action_space.n, save_dir=save_dir, checkpoint=checkpoint)

        # Create a logger for metrics
        logger = DDQNMetricLogger(save_dir)

        # Create a DDQN Trainer with curriculum
        trainer = ddqn_trainer.Trainer(agent=agent, env=teacher.games, episodes=total_episodes, logger=logger, curriculum=curriculum)

        trainer.train(render=render)

def main():
    """
    Main function for orchestrating different training scenarios.
    """
    total_episodes = 20_000
    render = False

    date = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M')

    # DDQN Baseline
    baseline_main(total_episodes=total_episodes, render=render)

    # DRDQN Baseline
    baseline_main(total_episodes=total_episodes, render=render, recurrent=True)

    # DDQN Movement Curriculum
    curriculum_experiment_main(total_episodes=total_episodes, date=date, basis='movement', render=render)

    # DDQN Version Curriculum
    curriculum_experiment_main(total_episodes=total_episodes, date=date, basis='version', render=render)

    # PPO
    ppo_trainer.train(total_episodes)

    # A3C
    a3c_trainer.train(total_episodes)

if __name__ == '__main__':
    main()
