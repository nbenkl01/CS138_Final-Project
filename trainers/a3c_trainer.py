import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
from environments.actor_critic_env import create_train_env_AC
from agents.ACnetworks import ActorCritic
import torch.nn.functional as F
import torch.multiprocessing as _mp
from torch.distributions import Categorical
from collections import deque
from tensorboardX import SummaryWriter
import timeit
import shutil
from utils.metrics import A3CMetricLogger

# Class to store options and hyperparameters for A3C training
class Options:
    def __init__(self):
        self.world=1
        self.stage=1
        self.action_type="simple"
        self.lr=1e-4
        self.gamma=0.9
        self.tau=1.0
        self.beta=0.01
        self.num_local_steps=50
        self.num_global_steps=5e6
        self.num_processes=6
        self.save_interval=50
        self.max_actionst=200
        self.log_path="tensorboard/a3c_super_mario_bros"
        self.saved_path="trained_models"
        self.load_from_previous_stage = False
        self.use_gpu = False

# Custom Adam optimizer for A3C with shared memory
class GlobalAdam(torch.optim.Adam):
    def __init__(self, params, lr):
        super(GlobalAdam, self).__init__(params, lr=lr)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

# Function to train the A3C model
def train(num_episodes):
    # Initialize options
    opt = Options()
    torch.manual_seed(42)
    
    # Clear existing log path and create a new one
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    
    # Create saved path directory if it doesn't exist
    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    
    # Set up multiprocessing context
    mp = _mp.get_context("spawn")
    
    # Create environment for training
    env, num_states, num_actions = create_train_env_AC(opt.world, opt.stage, opt.action_type)
    
    # Create global A3C model
    global_model = ActorCritic(num_states, num_actions)
    
    # Move model to GPU if available
    if opt.use_gpu:
        global_model.cuda()
    
    # Share model parameters among processes
    global_model.share_memory()
    
    # Load model from the previous stage if specified
    if opt.load_from_previous_stage:
        if opt.stage == 1:
            previous_world = opt.world - 1
            previous_stage = 4
        else:
            previous_world = opt.world
            previous_stage = opt.stage - 1
        file_ = "{}/a3c_super_mario_bros_{}_{}".format(opt.saved_path, previous_world, previous_stage)
        if os.path.isfile(file_):
            global_model.load_state_dict(torch.load(file_))

    # Create global optimizer
    optimizer = GlobalAdam(global_model.parameters(), lr=opt.lr)
    
    # List to store local processes
    processes = []
    
    # Start local training processes
    for index in range(opt.num_processes):
        if index == 0:
            process = mp.Process(target=local_train, args=(index, num_episodes, opt, global_model, optimizer, True))
        else:
            process = mp.Process(target=local_train, args=(index, num_episodes, opt, global_model, optimizer))
        process.start()
        processes.append(process)
    
    # Start local testing process
    process = mp.Process(target=local_test, args=(opt.num_processes, opt, global_model))
    process.start()
    processes.append(process)
    
    # Wait for all processes to finish
    for process in processes:
        process.join()

# Function for local training process
def local_train(index, num_episodes, opt, global_model, optimizer, save=False):
    # Initialize logger for metrics
    logger = A3CMetricLogger(opt.log_path)
    torch.manual_seed(42 + index)
    
    # Start timer for saving intervals
    if save:
        start_time = timeit.default_timer()
    
    # Create TensorBoard summary writer
    writer = SummaryWriter(opt.log_path)
    
    # Create local environment for training
    env, num_states, num_actions = create_train_env_AC(opt.world, opt.stage, opt.action_type)
    
    # Create local A3C model
    local_model = ActorCritic(num_states, num_actions)
    
    # Move model to GPU if available
    if opt.use_gpu:
        local_model.cuda()
    
    # Set model to training mode
    local_model.train()
    
    # Get the initial state
    state = torch.from_numpy(env.reset())
    
    # Move state to GPU if available
    if opt.use_gpu:
        state = state.cuda()
    
    # Initialize flags and counters
    done = True
    curr_step = 0
    curr_episode = 0
    
    # Training loop
    while curr_episode <= num_episodes:
        # Save the global model at regular intervals
        if save:
            if curr_episode % opt.save_interval == 0 and curr_episode > 0:
                torch.save(global_model.state_dict(),
                           "{}/a3c_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage))
            print("Process {}. Episode {}".format(index, curr_episode))
        
        # Increment episode count
        curr_episode += 1
        
        # Load global model parameters into the local model
        local_model.load_state_dict(global_model.state_dict())
        
        # Initialize hidden and cell states if the episode starts
        if done:
            h_0 = torch.zeros((1, 512), dtype=torch.float)
            c_0 = torch.zeros((1, 512), dtype=torch.float)
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()
        
        # Move hidden and cell states to GPU if available
        if opt.use_gpu:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
        
        # Lists to store data for each time step
        log_policies = []
        values = []
        rewards = []
        entropies = []

        # Local training loop
        for _ in range(opt.num_local_steps):
            curr_step += 1
            # Forward pass through the local model
            logits, value, h_0, c_0 = local_model(state, h_0, c_0)
            # Compute softmax policy and log policy
            policy = F.softmax(logits, dim=1)
            log_policy = F.log_softmax(logits, dim=1)
            # Calculate entropy of the policy
            entropy = -(policy * log_policy).sum(1, keepdim=True)
            
            # Create a Categorical distribution for sampling actions
            m = Categorical(policy)
            action = m.sample().item()
            
            # Take a step in the environment
            state, reward, done, _ = env.step(action)
            state = torch.from_numpy(state)
            
            # Move state to GPU if available
            if opt.use_gpu:
                state = state.cuda()
            
            # Terminate the episode if maximum global steps are reached
            if curr_step > opt.num_global_steps:
                done = True

            # Reset environment if the episode is done
            if done:
                curr_step = 0
                state = torch.from_numpy(env.reset())
                if opt.use_gpu:
                    state = state.cuda()

            # Store data for computing loss
            values.append(value)
            log_policies.append(log_policy[0, action])
            rewards.append(reward)
            entropies.append(entropy)

            # End episode if done
            if done:
                break

        # Compute returns and losses for each time step
        R = torch.zeros((1, 1), dtype=torch.float)
        if opt.use_gpu:
            R = R.cuda()
        if not done:
            _, R, _, _ = local_model(state, h_0, c_0)

        gae = torch.zeros((1, 1), dtype=torch.float)
        if opt.use_gpu:
            gae = gae.cuda()
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        next_value = R

        # Compute losses in reverse order
        for value, log_policy, reward, entropy in list(zip(values, log_policies, rewards, entropies))[::-1]:
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward + opt.gamma * next_value.detach() - value.detach()
            next_value = value
            actor_loss = actor_loss + log_policy * gae
            R = R * opt.gamma + reward
            critic_loss = critic_loss + (R - value) ** 2 / 2
            entropy_loss = entropy_loss + entropy
            total_loss = actor_loss + critic_loss - opt.beta * entropy_loss
            logger.log_step(reward, actor_loss, critic_loss, entropy_loss, total_loss)

        # Log episode statistics
        logger.log_episode()
        
        # Record metrics every 20 episodes
        if curr_episode % 20 == 0:
            logger.record(
                episode=curr_episode
            )
        
        # Calculate total loss and log to TensorBoard
        total_loss = -actor_loss + critic_loss - opt.beta * entropy_loss
        writer.add_scalar("Train_{}/Loss".format(index), total_loss, curr_episode)
        
        # Zero the gradients, perform backward pass, and update global model
        optimizer.zero_grad()
        total_loss.backward()

        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            if global_param.grad is not None:
                break
            global_param._grad = local_param.grad

        optimizer.step()

        # Check if the training process is complete
        if curr_episode == int(opt.num_global_steps / opt.num_local_steps):
            print("Training process {} terminated".format(index))
            if save:
                end_time = timeit.default_timer()
                print('The code runs for %.2f s ' % (end_time - start_time))
            return

# Function for local testing process
def local_test(index, opt, global_model):
    torch.manual_seed(42 + index)
    
    # Create local environment for testing
    env, num_states, num_actions = create_train_env_AC(opt.world, opt.stage, opt.action_type)
    
    # Create local A3C model for testing
    local_model = ActorCritic(num_states, num_actions)
    
    # Set model to evaluation mode
    local_model.eval()
    
    # Get the initial state
    state = torch.from_numpy(env.reset())
    
    # Initialize flags and counters
    done = True
    curr_step = 0
    actions = deque(maxlen=opt.max_actions)
    
    # Testing loop
    while True:
        curr_step += 1
        
        # Load the global model if a new episode starts
        if done:
            local_model.load_state_dict(global_model.state_dict())
        
        # No gradient computation during testing
        with torch.no_grad():
            # Initialize hidden and cell states if the episode starts
            if done:
                h_0 = torch.zeros((1, 512), dtype=torch.float)
                c_0 = torch.zeros((1, 512), dtype=torch.float)
            else:
                h_0 = h_0.detach()
                c_0 = c_0.detach()

        # Forward pass through the local model
        logits, value, h_0, c_0 = local_model(state, h_0, c_0)
        
        # Compute softmax policy and choose the action with the highest probability
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        
        # Take a step in the environment
        state, reward, done, _ = env.step(action)
        
        # Append the action to the action history
        actions.append(action)
        
        # End the episode if maximum global steps or a repetitive action sequence is reached
        if curr_step > opt.num_global_steps or actions.count(actions[0]) == actions.maxlen:
            done = True
        
        # Reset the environment if the episode is done
        if done:
            curr_step = 0
            actions.clear()
            state = env.reset()
        
        # Move state to PyTorch tensor
        state = torch.from_numpy(state)
