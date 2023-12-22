import os
os.environ['OMP_NUM_THREADS'] = '1'
import torch
from environments.actor_critic_env import create_train_env
from agents.ACnetworks import PPO
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import torch.nn.functional as F
import time, datetime
import pandas as pd
import numpy as np


def test():
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    saved_path = 'trained_models'
    actions = SIMPLE_MOVEMENT
    
    test_df = pd.DataFrame({'model':[],'level':[],'episode':[],'completion_time':[],'time':[], 'mean_reward':[]})
    for world in range(1,3):
        for level in range(1,5):
            env = create_train_env(world, level, actions)
            model = PPO(env.observation_space.shape[0], len(actions))
            if torch.cuda.is_available():
                model.load_state_dict(torch.load("{}/ppo_super_mario_bros_{}_{}".format(saved_path, 1, 1)))
                model.cuda()
            else:
                model.load_state_dict(torch.load("{}/ppo_super_mario_bros_{}_{}".format(saved_path, 1, 1),
                                                map_location=lambda storage, loc: storage))
            model.eval()
            state = torch.from_numpy(env.reset())
            
            for episode in range(10):
                last_record_time = time.time()
                rewards = []
                while True:
                    if torch.cuda.is_available():
                        state = state.cuda()
                    logits, value = model(state)
                    policy = F.softmax(logits, dim=1)
                    action = torch.argmax(policy).item()
                    state, reward, done, info = env.step(action)
                    rewards.append(reward)
                    state = torch.from_numpy(state)
                    # env.render()
                    if info["flag_get"]:
                        print("World {} stage {} completed".format(world, stage))
                        test_df = pd.concat([test_df, 
                                            pd.DataFrame({'model':['PPO'],
                                                        'world': [world],
                                                        'level':[level],
                                                        'success':[1],
                                                        'episode':[episode],
                                                        'final_x':[info['x_pos']],
                                                        'completion_time': [np.round(time.time() - last_record_time, 3)],
                                                        'time':[datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')],
                                                        'mean_reward':[np.mean(rewards)]})]).reset_index(drop=True)
                        break
                    elif done:
                        test_df = pd.concat([test_df, 
                                            pd.DataFrame({'model':['PPO'],
                                                        'world': [world],
                                                        'level':[level],
                                                        'success':[0],
                                                        'episode':[episode],
                                                        'final_x':[info['x_pos']],
                                                        'completion_time': [np.round(time.time() - last_record_time, 3)],
                                                        'time':[datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')],
                                                        'mean_reward':[np.mean(rewards)]})
                                            ]).reset_index(drop=True)
                        break
            test_df.to_csv('./tensorboard/ppo_super_mario_bros/eval.csv')
