import os
os.environ['OMP_NUM_THREADS'] = '1'
import torch
from environments.actor_critic_env import create_train_env_AC
from agents.ACnetworks import ActorCritic
import torch.nn.functional as F
import time, datetime
import pandas as pd
import numpy as np


def test():
    saved_path = 'trained_models'
    action_type = 'simple'
    torch.manual_seed(42)
    test_df = pd.DataFrame({'model':[],'level':[],'episode':[],'completion_time':[],'time':[], 'mean_reward':[]})
    for world in range(1,3):
        for level in range(1,5):
            env, num_states, num_actions = create_train_env_AC(world, level, action_type)
            model = ActorCritic(num_states, num_actions)
            if torch.cuda.is_available():
                model.load_state_dict(torch.load("{}/a3c_super_mario_bros_{}_{}".format(saved_path, 1, 1)))
                model.cuda()
            else:
                model.load_state_dict(torch.load("{}/a3c_super_mario_bros_{}_{}".format(saved_path, 1, 1),
                                                map_location=lambda storage, loc: storage))
            model.eval()
            state = torch.from_numpy(env.reset())
            done = True
            for episode in range(10):
                print(f'episode: {episode}')
                last_record_time = time.time()
                rewards = []
                while True:
                    if done:
                        h_0 = torch.zeros((1, 512), dtype=torch.float)
                        c_0 = torch.zeros((1, 512), dtype=torch.float)
                        env.reset()
                        done = False
                    else:
                        h_0 = h_0.detach()
                        c_0 = c_0.detach()
                    if torch.cuda.is_available():
                        h_0 = h_0.cuda()
                        c_0 = c_0.cuda()
                        state = state.cuda()

                    logits, value, h_0, c_0 = model(state, h_0, c_0)
                    policy = F.softmax(logits, dim=1)
                    action = torch.argmax(policy).item()
                    action = int(action)
                    state, reward, done, info = env.step(action)
                    rewards.append(reward)
                    state = torch.from_numpy(state)
                    # env.render()
                    if info["flag_get"]:
                        print("World {} stage {} completed".format(world, level))
                        test_df = pd.concat([test_df, 
                                            pd.DataFrame({'model':['A3C'],
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
                                            pd.DataFrame({'model':['A3C'],
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
            test_df.to_csv('./tensorboard/a3c_super_mario_bros/eval.csv')