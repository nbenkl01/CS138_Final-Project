import torch.nn as nn
import torch.nn.functional as F

# Define the Proximal Policy Optimization (PPO) network
class PPO(nn.Module):
    def __init__(self, num_inputs, num_actions):
        """
        Initialize the PPO network.

        Parameters:
        - num_inputs (int): Number of input channels.
        - num_actions (int): Number of possible actions.
        """
        super(PPO, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        # Fully connected layers
        self.linear = nn.Linear(32 * 6 * 6, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights for convolutional and linear layers.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        Forward pass through the PPO network.

        Parameters:
        - x (tensor): Input tensor.

        Returns:
        - actor_output, critic_output (tensors): Output of actor and critic networks.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.linear(x.view(x.size(0), -1))
        return self.actor_linear(x), self.critic_linear(x)

# Define the Actor-Critic network
class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions):
        """
        Initialize the Actor-Critic network.

        Parameters:
        - num_inputs (int): Number of input channels.
        - num_actions (int): Number of possible actions.
        """
        super(ActorCritic, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        # Fully connected layers
        self.linear = nn.Linear(32 * 6 * 6, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights for convolutional and linear layers.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x, hx, cx):
        """
        Forward pass through the Actor-Critic network.

        Parameters:
        - x (tensor): Input tensor.
        - hx (tensor): Hidden state of LSTM.
        - cx (tensor): Cell state of LSTM.

        Returns:
        - actor_output, critic_output, new_hx, new_cx (tensors): Output of actor and critic networks,
          and updated hidden and cell states of LSTM.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        hx, cx = self.linear(x.view(x.size(0), -1), (hx, cx))
        return self.actor_linear(hx), self.critic_linear(hx), hx, cx