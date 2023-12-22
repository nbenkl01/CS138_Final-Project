import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from environments.wrappers import ResizeObservation, SkipFrame

# Class to manage the Super Mario game environment with various wrappers
class SuperMarioGame:
    def __init__(self, world=1, stage=1, version=0, movement=RIGHT_ONLY):
        """
        Initialize the SuperMarioGame class.

        Parameters:
        - world (int): World number.
        - stage (int): Stage number.
        - version (int): Environment version.
        - movement (list): List of allowed movements/actions.
        """
        # Create the Super Mario Bros environment
        self.env = gym_super_mario_bros.make(f'SuperMarioBros-{world}-{stage}-v{version}')
        # Apply JoypadSpace wrapper to limit available movements
        self.env = JoypadSpace(self.env, movement)

    def transform(self, skip=4, gray=True, resize=84, transform=255, stack=4):
        """
        Transform the environment using various wrappers.

        Parameters:
        - skip (int): Number of frames to skip.
        - gray (bool): Convert frames to grayscale.
        - resize (int): Resize the frames to the specified shape.
        - transform (int): Value to normalize the pixel values.
        - stack (int): Number of frames to stack.

        Returns:
        - env: Transformed environment.
        """
        env = self.env
        # Apply SkipFrame wrapper to skip frames
        if skip:
            env = SkipFrame(env, skip=skip)
        # Apply GrayScaleObservation wrapper to convert frames to grayscale
        if gray:
            env = GrayScaleObservation(env, keep_dim=False)
        # Apply ResizeObservation wrapper to resize frames
        if resize:
            env = ResizeObservation(env, shape=resize)
        # Apply TransformObservation wrapper to normalize pixel values
        if transform:
            env = TransformObservation(env, f=lambda x: x / transform)
        # Apply FrameStack wrapper to stack frames
        if stack:
            env = FrameStack(env, num_stack=stack)
        self.env = env
        return env
