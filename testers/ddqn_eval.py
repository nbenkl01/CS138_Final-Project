import datetime
from pathlib import Path

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace

from utils.metrics import DDQNMetricLogger
from agents.Qagent import Mario
from environments.wrappers import ResizeObservation, SkipFrame
import os
import time, datetime
import pandas as pd
import numpy as np

def test():
    test_df = pd.DataFrame({'model':[],'level':[],'episode':[],'completion_time':[], 'mean_reward':[],'time':[]})
    for model in os.listdir(f'./final_models/'):
        if model.endswith('.chkpt'):
            checkpoint = Path(f'./final_models/{model}')
            for world in range(1,3):
                for level in range(1,5):
                    env = gym_super_mario_bros.make(f'SuperMarioBros-{world}-{level}-v0')

                    if model.startswith('movement'):
                        env = JoypadSpace(
                            env,
                            SIMPLE_MOVEMENT + [['left', 'A'],
                                                ['left', 'B'],
                                                ['left', 'A', 'B']]
                            )
                    else:
                        env = JoypadSpace(
                            env,
                            SIMPLE_MOVEMENT
                            )

                    env = SkipFrame(env, skip=4)
                    env = GrayScaleObservation(env, keep_dim=False)
                    env = ResizeObservation(env, shape=84)
                    env = TransformObservation(env, f=lambda x: x / 255)
                    env = FrameStack(env, num_stack=4)
                    env.reset()

                    save_dir = Path(f'final_models/') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
                    save_dir.mkdir(parents=True)

                    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)
                    mario.exploration_rate = mario.exploration_rate_min

                    logger = DDQNMetricLogger(save_dir)

                    episodes = 10

                    step = 0
                    last_record_time = time.time()
                    rewards = []
                    for e in range(episodes):

                        state = env.reset()

                        while True:
                            step += 1
                            action = mario.act(state)

                            next_state, reward, done, info = env.step(action)
                            rewards.append(reward)

                            mario.cache(state, next_state, action, reward, done)

                            logger.log_step(reward, None, None)

                            state = next_state

                            if info['flag_get']:
                                test_df = pd.concat([test_df, 
                                                    pd.DataFrame({'model':[model.split('.')[0]],
                                                                'world': [world],
                                                                'level':[level],
                                                                'success':[1],
                                                                'final_x':[info['x_pos']],
                                                                'episode':[e],
                                                                'completion_time': [np.round(time.time() - last_record_time, 3)],
                                                                'time':[datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')],
                                                                'mean_reward':[np.mean(rewards)]})
                                                    ]).reset_index(drop=True)
                                print("Episode: {0},\tSteps: {1},\tscore: {2}"
                                    .format(e, step, np.mean(rewards))
                                )
                                break
                            elif done:
                                test_df = pd.concat([test_df, 
                                                    pd.DataFrame({'model':[model.split('.')[0]],
                                                                'world': [world],
                                                                'level':[level],
                                                                'success':[0],
                                                                'final_x':[info['x_pos']],
                                                                'episode':[e],
                                                                'completion_time': [np.round(time.time() - last_record_time, 3)],
                                                                'time':[datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')],
                                                                'mean_reward':[np.mean(rewards)]})
                                                    ]).reset_index(drop=True)
                                print("Episode: {0},\tSteps: {1},\tscore: {2}"
                                    .format(e, step, np.mean(rewards))
                                )
                                break

                        logger.log_episode()

                        if e % 20 == 0:
                            logger.record(
                                episode=e,
                                epsilon=mario.exploration_rate,
                                step=mario.curr_step
                            )
                    env.close()
                    test_df.to_csv(f'./final_models/eval.csv')