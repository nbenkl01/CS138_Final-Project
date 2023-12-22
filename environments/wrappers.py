import gym
from gym import Wrapper
import numpy as np
from skimage import transform

from gym.spaces import Box
import cv2
import numpy as np

# QNetwork Wrappers

# Wrapper to resize the observation
class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        """
        Initialize the ResizeObservation wrapper.

        Parameters:
        - env (gym.Env): The environment to wrap.
        - shape (int or tuple): The desired shape of the observation after resizing.
        """
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        """
        Preprocess the observation by resizing it.

        Parameters:
        - observation (np.ndarray): The input observation.

        Returns:
        - np.ndarray: The resized observation.
        """
        resize_obs = transform.resize(observation, self.shape)
        resize_obs *= 255
        resize_obs = resize_obs.astype(np.uint8)
        return resize_obs

# Wrapper to skip frames
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """
        Initialize the SkipFrame wrapper.

        Parameters:
        - env (gym.Env): The environment to wrap.
        - skip (int): The number of frames to skip between returned frames.
        """
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """
        Repeat the action, and sum the rewards over skipped frames.

        Parameters:
        - action: The action to take.

        Returns:
        - tuple: The next observation, total reward, done flag, and additional info.
        """
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info
    
# Function to process a frame, converting it to grayscale, resizing, and normalizing
def process_frame(frame):
    """
    Process a frame by converting it to grayscale, resizing, and normalizing.

    Parameters:
    - frame (np.ndarray): The input frame.

    Returns:
    - np.ndarray: The processed frame.
    """
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
        return frame
    else:
        return np.zeros((1, 84, 84))

# Actor-Critic Wrappers

# Wrapper to customize the reward
class CustomReward(Wrapper):
    def __init__(self, env=None, world=None, stage=None):
        """
        Initialize the CustomReward wrapper.

        Parameters:
        - env (gym.Env): The environment to wrap.
        - world (optional): Additional parameters for customization.
        - stage (optional): Additional parameters for customization.
        """
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0
        self.current_x = 40
        self.world = world
        self.stage = stage

    def step(self, action):
        """
        Modify the step function to include customization of rewards.

        Parameters:
        - action: The action to take.

        Returns:
        - tuple: The next observation, modified reward, done flag, and additional info.
        """
        state, reward, done, info = self.env.step(action)
        state = process_frame(state)

        self.current_x = info["x_pos"]
        return state, reward, done, info

    def reset(self):
        """
        Reset the environment, including resetting customization parameters.

        Returns:
        - np.ndarray: The reset observation.
        """
        self.curr_score = 0
        self.current_x = 40
        return process_frame(self.env.reset())
    
# Wrapper to skip frames with custom state handling
class CustomSkipFrame(Wrapper):
    def __init__(self, env, skip=4):
        """
        Initialize the CustomSkipFrame wrapper.

        Parameters:
        - env (gym.Env): The environment to wrap.
        - skip (int): The number of frames to skip between returned frames.
        """
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(skip, 84, 84))
        self.skip = skip
        self.states = np.zeros((skip, 84, 84), dtype=np.float32)

    def step(self, action):
        """
        Repeat the action over skipped frames and handle the state accordingly.

        Parameters:
        - action: The action to take.

        Returns:
        - tuple: The next observation, total reward, done flag, and additional info.
        """
        total_reward = 0
        last_states = []
        for i in range(self.skip):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            if i >= self.skip / 2:
                last_states.append(state)
            if done:
                self.reset()
                return self.states[None, :, :, :].astype(np.float32), total_reward, done, info
        max_state = np.max(np.concatenate(last_states, 0), 0)
        self.states[:-1] = self.states[1:]
        self.states[-1] = max_state
        return self.states[None, :, :, :].astype(np.float32), total_reward, done, info

    def reset(self):
        """
        Reset the environment and states.

        Returns:
        - np.ndarray: The reset observation.
        """
        state = self.env.reset()
        self.states = np.concatenate([state for _ in range(self.skip)], 0)
        return self.states[None, :, :, :].astype(np.float32)