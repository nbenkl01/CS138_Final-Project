import torch
import random
import numpy as np
from Qnetworks import DDQN, DRDQN
from collections import deque

class Mario:
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=None, recurrent=False):
        """
        Initialize the Mario agent.

        Parameters:
        - state_dim (tuple): Input dimensions (channels, height, width) of the state.
        - action_dim (int): Number of possible actions.
        - save_dir (str): Directory to save the trained model.
        - checkpoint (str): Path to a checkpoint file for model loading (default=None).
        - recurrent (bool): Flag indicating whether to use a recurrent network (default=False).
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=100000)
        self.batch_size = 32

        # Exploration parameters
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.999999
        self.exploration_rate_min = 0.1
        self.gamma = 0.9

        # Training parameters
        self.curr_step = 0
        self.burnin = 1e5  # Minimum experiences before training
        self.learn_every = 3  # Number of experiences between updates to Q_online
        self.sync_every = 1e4  # Number of experiences between Q_target & Q_online sync

        # Saving parameters
        self.save_every = 5e5  # Number of experiences between saving Mario Net
        self.save_dir = save_dir

        # CUDA availability check
        self.use_cuda = torch.cuda.is_available()

        # Define Mario's DNN to predict the most optimal action
        if recurrent:
            self.net = DRDQN(self.state_dim, self.action_dim).float()
        else:
            self.net = DDQN(self.state_dim, self.action_dim).float()

        # Move model to CUDA if available
        if self.use_cuda:
            self.net = self.net.to(device='cuda')

        # Load model from a checkpoint if provided
        if checkpoint:
            self.load(checkpoint)

        # Define optimizer and loss function
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def act(self, state):
        """
        Choose an epsilon-greedy action and update the exploration rate.

        Parameters:
        - state (LazyFrame): A single observation of the current state.

        Returns:
        - action_idx (int): Index of the chosen action.
        """
        # Explore
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        # Exploit
        else:
            state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model='online')
            action_idx = torch.argmax(action_values, axis=1).item()

        # Decrease exploration rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # Increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience in the replay buffer.

        Parameters:
        - state (LazyFrame): Current state.
        - next_state (LazyFrame): Next state.
        - action (int): Chosen action.
        - reward (float): Received reward.
        - done (bool): Whether the episode is done.
        """
        state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state).cuda() if self.use_cuda else torch.FloatTensor(next_state)
        action = torch.LongTensor([action]).cuda() if self.use_cuda else torch.LongTensor([action])
        reward = torch.DoubleTensor([reward]).cuda() if self.use_cuda else torch.DoubleTensor([reward])
        done = torch.BoolTensor([done]).cuda() if self.use_cuda else torch.BoolTensor([done])

        self.memory.append((state, next_state, action, reward, done,))

    def recall(self):
        """
        Retrieve a batch of experiences from memory.

        Returns:
        - state, next_state, action, reward, done (tensors): Batch of experiences.
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        """
        Calculate the temporal difference estimate.

        Parameters:
        - state (tensor): Current state.
        - action (tensor): Chosen action.

        Returns:
        - current_Q (tensor): Q-value for the current state-action pair.
        """
        current_Q = self.net(state, model='online')[np.arange(0, self.batch_size), action]  # Q_online(s, a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        """
        Calculate the temporal difference target.

        Parameters:
        - reward (tensor): Received reward.
        - next_state (tensor): Next state.
        - done (tensor): Whether the episode is done.

        Returns:
        - td_target (tensor): Temporal difference target.
        """
        next_state_Q = self.net(next_state, model='online')
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model='target')[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        """
        Update the online Q-network.

        Parameters:
        - td_estimate (tensor): Temporal difference estimate.
        - td_target (tensor): Temporal difference target.

        Returns:
        - loss.item() (float): Loss value.
        """
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        """Synchronize the target Q-network with the online Q-network."""
        self.net.target.load_state_dict(self.net.online.state_dict())

    def learn(self):
        """
        Update the Q-network based on the stored experiences.

        Returns:
        - (td_estimate.mean().item(), loss) (tuple): Tuple containing TD estimate mean and loss values.
        """
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        state, next_state, action, reward, done = self.recall()

        td_est = self.td_estimate(state, action)

        td_tgt = self.td_target(reward, next_state, done)

        loss = self.update_Q_online(td_est, td_tgt)

        return td_est.mean().item(), loss

    def save(self):
        """Save the current model and exploration rate."""
        save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=self.exploration_rate
            ),
            save_path
        )

    def load(self, load_path):
        """
        Load a model from a checkpoint file.

        Parameters:
        - load_path (str): Path to the checkpoint file.
        """
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate
