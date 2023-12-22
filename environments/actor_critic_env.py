import gym_super_mario_bros
from environments.wrappers import CustomReward, CustomSkipFrame
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import cv2
import numpy as np
import torch.multiprocessing as mp

# Function to process a frame, converting it to grayscale, resizing, and normalizing
def process_frame(frame):
    """
    Preprocesses a frame by converting it to grayscale, resizing, and normalizing.

    Parameters:
    - frame: Input frame in RGB format.

    Returns:
    - processed_frame: Preprocessed frame.
    """
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
        return frame
    else:
        return np.zeros((1, 84, 84))

# Function to create a training environment with custom wrappers
def create_train_env(world, stage, actions):
    """
    Creates a training environment with custom wrappers.

    Parameters:
    - world (int): World number.
    - stage (int): Stage number.
    - actions: List of allowed movements/actions.

    Returns:
    - env: Training environment.
    """
    env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage))

    env = JoypadSpace(env, actions)
    env = CustomReward(env, world, stage)
    env = CustomSkipFrame(env)
    return env

# Function to create a training environment for Actor-Critic with custom wrappers
def create_train_env_AC(world, stage, action_type):
    """
    Creates a training environment for Actor-Critic with custom wrappers.

    Parameters:
    - world (int): World number.
    - stage (int): Stage number.
    - action_type (str): Type of actions to use.

    Returns:
    - env: Training environment.
    - num_states (int): Number of states in the observation space.
    - num_actions (int): Number of available actions.
    """
    env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage))
    if action_type == "right":
        actions = RIGHT_ONLY
    elif action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT
    env = JoypadSpace(env, actions)
    env = CustomSkipFrame(env)
    return env, env.observation_space.shape[0], len(actions)

# Class to manage multiple environments in parallel
class MultipleEnvironments:
    def __init__(self, world, stage, action_type, num_envs):
        """
        Initializes the MultipleEnvironments class.

        Parameters:
        - world (int): World number.
        - stage (int): Stage number.
        - action_type (str): Type of actions to use.
        - num_envs (int): Number of parallel environments.
        """
        # Create communication pipes between agent and environments
        self.agent_conns, self.env_conns = zip(*[mp.Pipe() for _ in range(num_envs)])
        # Determine available actions based on the specified action_type
        if action_type == "right":
            actions = RIGHT_ONLY
        elif action_type == "simple":
            actions = SIMPLE_MOVEMENT
        else:
            actions = COMPLEX_MOVEMENT
        # Create a list of training environments with custom wrappers
        self.envs = [create_train_env(world, stage, actions) for _ in range(num_envs)]
        self.num_states = self.envs[0].observation_space.shape[0]
        self.num_actions = len(actions)
        # Start separate processes for each environment
        for index in range(num_envs):
            process = mp.Process(target=self.run, args=(index,))
            process.start()
            self.env_conns[index].close()

    # Function to run an environment process
    def run(self, index):
        """
        Runs an environment process for parallel execution.

        Parameters:
        - index (int): Index of the environment process.
        """
        self.agent_conns[index].close()
        while True:
            # Receive requests and actions from the agent
            request, action = self.env_conns[index].recv()
            if request == "step":
                # Perform a step in the environment and send the result back to the agent
                self.env_conns[index].send(self.envs[index].step(action.item()))
            elif request == "reset":
                # Reset the environment and send the initial observation back to the agent
                self.env_conns[index].send(self.envs[index].reset())
            else:
                raise NotImplementedError
