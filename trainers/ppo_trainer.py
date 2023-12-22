import os

import torch
from environments.actor_critic_env import create_train_env, MultipleEnvironments
from agents.ACnetworks import PPO
import torch.nn.functional as F
import torch.multiprocessing as _mp
from torch.distributions import Categorical
from collections import deque
from tensorboardX import SummaryWriter
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import numpy as np
import shutil
from utils.metrics import PPOMetricLogger

# Class to store options and hyperparameters for training
class Options:
    def __init__(self):
        self.world=1
        self.stage=1
        self.action_type="simple"
        self.lr=1e-4
        self.gamma=0.9
        self.tau=1.0
        self.beta=0.01
        self.epsilon=0.2
        self.batch_size=16
        self.num_epochs=10
        self.num_local_steps=512
        self.num_global_steps=5e6
        self.num_processes=8
        self.save_interval=50
        self.max_actions=200
        self.log_path="tensorboard/ppo_super_mario_bros"
        self.saved_path="trained_models"

# Function to train the PPO model
def train(num_episodes):
    # Initialize options
    opt = Options()
    
    # Set seed for reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)
    
    # Clear existing log path and create a new one
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    
    # Initialize PPO metric logger
    logger = PPOMetricLogger(opt.log_path)
    
    # Create saved path directory if it doesn't exist
    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    
    # Set up multiprocessing context
    mp = _mp.get_context("spawn")
    
    # Create multiple environments for training
    envs = MultipleEnvironments(opt.world, opt.stage, opt.action_type, opt.num_processes)
    
    # Create PPO model
    model = PPO(envs.num_states, envs.num_actions)
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model.cuda()
    
    # Share model parameters among processes
    model.share_memory()
    
    # Start a separate process for evaluation
    process = mp.Process(target=eval, args=(opt, model, envs.num_states, envs.num_actions))
    process.start()
    
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    
    # Send reset signal to all environments
    [agent_conn.send(("reset", None)) for agent_conn in envs.agent_conns]
    
    # Receive initial states from environments
    curr_states = [agent_conn.recv() for agent_conn in envs.agent_conns]
    curr_states = torch.from_numpy(np.concatenate(curr_states, 0))
    
    # Move current states to GPU if available
    if torch.cuda.is_available():
        curr_states = curr_states.cuda()
    
    # Initialize episode count
    curr_episode = 0
    
    # Training loop
    while curr_episode <= num_episodes:
        curr_episode += 1
        
        # Lists to store information for each step
        old_log_policies = []
        actions = []
        values = []
        states = []
        rewards = []
        dones = []
        
        # Collect data for a certain number of local steps
        for _ in range(opt.num_local_steps):
            states.append(curr_states)
            logits, value = model(curr_states)
            values.append(value.squeeze())
            policy = F.softmax(logits, dim=1)
            old_m = Categorical(policy)
            action = old_m.sample()
            actions.append(action)
            old_log_policy = old_m.log_prob(action)
            old_log_policies.append(old_log_policy)
            
            # Send actions to environments and receive results
            if torch.cuda.is_available():
                [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action.cpu())]
            else:
                [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action)]

            state, reward, done, info = zip(*[agent_conn.recv() for agent_conn in envs.agent_conns])
            state = torch.from_numpy(np.concatenate(state, 0))
            
            # Move state, reward, and done to GPU if available
            if torch.cuda.is_available():
                state = state.cuda()
                reward = torch.cuda.FloatTensor(reward)
                done = torch.cuda.FloatTensor(done)
            else:
                reward = torch.FloatTensor(reward)
                done = torch.FloatTensor(done)
            
            # Log step information
            logger.log_step(reward, None, None, None, None)
            
            rewards.append(reward)
            dones.append(done)
            curr_states = state

        # Get value of the next state
        _, next_value, = model(curr_states)
        next_value = next_value.squeeze()
        
        # Prepare data for PPO update
        old_log_policies = torch.cat(old_log_policies).detach()
        actions = torch.cat(actions)
        values = torch.cat(values).detach()
        states = torch.cat(states)
        
        # Generalized Advantage Estimation (GAE) calculation
        gae = 0
        R = []
        for value, reward, done in list(zip(values, rewards, dones))[::-1]:
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward + opt.gamma * next_value.detach() * (1 - done) - value.detach()
            next_value = value
            R.append(gae + value)
        R = R[::-1]
        R = torch.cat(R).detach()
        advantages = R - values
        
        # PPO update loop
        for i in range(opt.num_epochs):
            # Randomly shuffle data indices
            indice = torch.randperm(opt.num_local_steps * opt.num_processes)
            
            # Mini-batch loop
            for j in range(opt.batch_size):
                # Extract batch indices for the mini-batch
                batch_indices = indice[
                    int(j * (opt.num_local_steps * opt.num_processes / opt.batch_size)): int((j + 1) * (
                            opt.num_local_steps * opt.num_processes / opt.batch_size))]
                
                # Forward pass for the mini-batch
                logits, value = model(states[batch_indices])
                new_policy = F.softmax(logits, dim=1)
                new_m = Categorical(new_policy)
                new_log_policy = new_m.log_prob(actions[batch_indices])
                
                # PPO objective function
                ratio = torch.exp(new_log_policy - old_log_policies[batch_indices])
                actor_loss = -torch.mean(torch.min(ratio * advantages[batch_indices],
                                                   torch.clamp(ratio, 1.0 - opt.epsilon, 1.0 + opt.epsilon) *
                                                   advantages[batch_indices]))
                critic_loss = F.smooth_l1_loss(R[batch_indices], value.squeeze())
                entropy_loss = torch.mean(new_m.entropy())
                total_loss = actor_loss + critic_loss - opt.beta * entropy_loss
                
                # Log losses for monitoring
                logger.log_step(None, actor_loss, critic_loss, entropy_loss, total_loss)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        # Log episode statistics
        logger.log_episode()
        
        # Print episode information
        print("Episode: {}. Total loss: {}".format(curr_episode, total_loss))
        
        # Record metrics every 20 episodes
        if curr_episode % 20 == 0:
            logger.record(
                episode=curr_episode
            )

# Function for evaluating the trained model
def eval(opt, global_model, num_states, num_actions):
    torch.manual_seed(42)
    
    # Determine the action space based on the action type
    if opt.action_type == "right":
        actions = RIGHT_ONLY
    elif opt.action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT
    
    # Create a single evaluation environment
    env = create_train_env(opt.world, opt.stage, actions)
    
    # Create a local model for evaluation
    local_model = PPO(num_states, num_actions)
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        local_model.cuda()
    
    # Set model to evaluation mode
    local_model.eval()
    
    # Get the initial state
    state = torch.from_numpy(env.reset())
    
    # Move state to GPU if available
    if torch.cuda.is_available():
        state = state.cuda()
    
    # Initialize flags and counters
    done = True
    curr_step = 0
    actions = deque(maxlen=opt.max_actions)
    
    # Evaluation loop
    while True:
        curr_step += 1
        
        # Load the global model if a new episode starts
        if done:
            local_model.load_state_dict(global_model.state_dict())
        
        # Forward pass for the local model
        logits, value = local_model(state)
        policy = F.softmax(logits, dim=1)
        
        # Choose the action with the highest probability
        action = torch.argmax(policy).item()
        
        # Take a step in the environment
        state, reward, done, info = env.step(action)
        
        # Append the action to the action history
        actions.append(action)
        
        # If the agent has taken the same action for a number of consecutive steps or reached the maximum global steps, end the episode
        if curr_step > opt.num_global_steps or actions.count(actions[0]) == actions.maxlen:
            done = True
        
        # Reset the environment if the episode is done
        if done:
            curr_step = 0
            actions.clear()
            state = env.reset()
        
        # Move state to PyTorch tensor
        state = torch.from_numpy(state)
        
        # Move state to GPU if available
        if torch.cuda.is_available():
            state = state.cuda()