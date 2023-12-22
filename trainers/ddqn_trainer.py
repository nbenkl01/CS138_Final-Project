import random
import numpy as np

from gym_super_mario_bros.actions import RIGHT_ONLY
from environments.qlearning_env import SuperMarioGame

class Trainer:
    def __init__(self, agent, env, episodes, logger, curriculum=False):
        """
        Trainer class to facilitate training the agent.

        Parameters:
        - agent: The learning agent
        - env: The environment for training
        - episodes: Number of episodes to train
        - logger: Logger for recording metrics
        - curriculum: Whether to use a curriculum for training
        """
        if (type(env) == list) and curriculum:
            self.env = env[0]
        else:
            self.env = env

        self.agent = agent
        self.episodes = len(curriculum) if curriculum else episodes
        self.logger = logger
        self.curriculum = curriculum

    def train(self, render=False):
        """
        Train the agent using the specified environment.

        Parameters:
        - render: Whether to render the environment during training
        """
        env = self.env
        env.reset()

        agent = self.agent

        for e in range(self.episodes):
            if self.curriculum:
                env = self.curriculum[e]

            state = env.reset()

            # Play the game!
            while True:
                if render:
                    env.render()  # Render the environment

                action = agent.act(state)  # Agent chooses an action

                next_state, reward, done, info = env.step(action)  # Environment interaction

                agent.cache(state, next_state, action, reward, done)  # Remember the experience

                q, loss = agent.learn()  # Agent learns from the experience

                self.logger.log_step(reward, loss, q)  # Log the step metrics

                state = next_state  # Update the state

                if done or info['flag_get']:  # Check if the game is over
                    break

            self.logger.log_episode()  # Log the episode metrics

            if e % 20 == 0:
                self.logger.record(
                    episode=e,
                    epsilon=agent.exploration_rate,
                    step=agent.curr_step
                )

class Curriculum:
    def __init__(self, n_stages=2, curriculum_type='baseline'):
        """
        Curriculum class to generate a curriculum for training.

        Parameters:
        - n_stages: Number of stages in the curriculum
        - curriculum_type: Type of curriculum to generate
        """
        self.n_stages = n_stages
        self.curriculum_type = curriculum_type

    def gen_curriculum_baseline(self):
        """
        Generate a baseline curriculum with equal distribution.

        Returns:
        - List: Baseline curriculum distribution
        """
        return [[1 / self.n_stages for _ in range(self.n_stages)]]

    def gen_curriculum(self):
        """
        Generate the curriculum based on the specified type.

        Returns:
        - List: Curriculum distribution
        """
        if self.curriculum_type.lower().startswith('b'):
            self.curriculum_dist = self.gen_curriculum_baseline()
        elif self.curriculum_type.lower().startswith('n'):
            self.curriculum_dist = [[1 if i == j else 0 for j in range(self.n_stages)] for i in range(self.n_stages)] + self.gen_curriculum_baseline()
        elif self.curriculum_type.lower().startswith('m'):
            self.curriculum_dist = [[1 / (i + 1) if j <= i else 0 for j in range(self.n_stages)] for i in range(self.n_stages)]
        elif self.curriculum_type.lower().startswith('c'):
            self.curriculum_dist = [[1 / (2 * (i + 1)) if j < i else 1 / 2 + 1 / (2 * (i + 1)) if i == j else 0 for j in range(self.n_stages)] for i in range(self.n_stages)] + self.gen_curriculum_baseline()
        return self.curriculum_dist

    def gen_envs(self, envs, num_episodes, seed=42):
        """
        Generate environments based on the curriculum.

        Parameters:
        - envs: List of environments
        - num_episodes: Number of episodes per environment
        - seed: Seed for randomization

        Returns:
        - List: Curriculum environments
        """
        random.seed(seed)
        indecies = [i for i in range(len(envs))]
        pf = [random.choices(indecies, weights=dist, k=num_episodes) for dist in self.curriculum_dist]
        self.curriculum_indecies = np.array(pf).flatten()
        self.curriculum = [envs[i] for i in self.curriculum_indecies]
        return self.curriculum

class CurriculumTeacher:
    def __init__(self):
        self.movements = [RIGHT_ONLY]
        self.versions = [0]

    def movement_curriculum(self):
        """
        Define movement curricula for different stages.
        """
        movement_stage1 = [
            ['NOOP'],
            ['right'],
            ['right', 'A'],
        ] + [['NOOP']] * 7
        movement_stage3 = [
            ['NOOP'],
            ['right'],
            ['right', 'A'],
            ['NOOP'],
            ['NOOP'],
            ['NOOP'],
            ['left'],
            ['left', 'A'],
        ] + [['NOOP']] * 2
        movement_stage4 = [['NOOP'],
                           ['right'],
                           ['right', 'A'],
                           ['right', 'B'],
                           ['right', 'A', 'B'],
                           ['A'],
                           ['left'],
                           ['left', 'A'],
                           ['left', 'B'],
                           ['left', 'A', 'B']]
        self.movements = [movement_stage1,
                          movement_stage3,
                          movement_stage4]

    def version_curriculum(self):
        """
        Define version curricula.
        """
        self.versions = [3, 1, 0]

    def gen_environments(self):
        """
        Generate environments based on movement and version curricula.
        """
        games = []
        for version, movement in [(v, m) for m in self.movements for v in self.versions]:
            game = SuperMarioGame(world=1, stage=1, version=version,
                                  movement=movement)
            env = game.transform(skip=4,
                                 gray=True,
                                 resize=84,
                                 transform=255,
                                 stack=4)
            env.reset()
            games.append(env)
        self.games = games

    def gen_curricula(self, num_total_episodes, curriculum_type):
        """
        Generate curricula based on the specified type.

        Parameters:
        - num_total_episodes: Total number of episodes
        - curriculum_type: Type of curriculum

        Returns:
        - List: Curriculum environments
        """
        curriculum_gen = Curriculum(n_stages=len(self.games), curriculum_type=curriculum_type)
        curriculum_gen.gen_curriculum()
        episodes_per_dist = int(np.floor(num_total_episodes / len(self.games)))
        curriculum = curriculum_gen.gen_envs(envs=self.games, num_episodes=episodes_per_dist)
        self.curriculum = curriculum
        return curriculum


def curriculum_experiment(total_episodes=4000, curriculum_basis='movement'):
    """
    Conduct a curriculum experiment.

    Parameters:
    - total_episodes: Total number of episodes
    - curriculum_basis: Basis for curriculum generation

    Returns:
    - Dictionary: Curricula based on different types
    - CurriculumTeacher: Curriculum teacher object
    """
    teacher = CurriculumTeacher()
    if any([basis.lower().startswith('m') for basis in curriculum_basis.split('_')]):
        teacher.movement_curriculum()
    if any([basis.lower().startswith('v') for basis in curriculum_basis.split('_')]):
        teacher.version_curriculum()
    teacher.gen_environments()
    curricula = {}
    for curr_type in ['naive']:
        curricula[curr_type] = teacher.gen_curricula(num_total_episodes=total_episodes,
                                                      curriculum_type=curr_type)
    return curricula, teacher
