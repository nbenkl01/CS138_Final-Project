# Import necessary modules
from torch import nn
import copy
from collections import deque

# Define a Double Deep Q Network (DDQN) class
class DDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Initialize the DDQN model.

        Parameters:
        - input_dim (tuple): Input dimensions (channels, height, width) of the state.
        - output_dim (int): Number of actions in the output layer.
        """
        super().__init__()
        c, h, w = input_dim

        # Define the online network architecture
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        # Create a target network by deep copying the online network
        self.target = copy.deepcopy(self.online)

        # Freeze the parameters of the target network
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        """
        Forward pass through the DDQN model.

        Parameters:
        - input (tensor): Input state tensor.
        - model (str): Specifies whether to use 'online' or 'target' network.

        Returns:
        - tensor: Output Q-values.
        """
        # Specify whether to use the online or target network during forward pass
        if model == 'online':
            return self.online(input)
        elif model == 'target':
            return self.target(input)

# Define a Deep Recurrent Double Q Network (DRDQN) class
class DRDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Initialize the DRDQN model.

        Parameters:
        - input_dim (tuple): Input dimensions (channels, height, width) of the state.
        - output_dim (int): Number of actions in the output layer.
        """
        super().__init__()
        c, h, w = input_dim

        # Check if the input height and width match the expected values
        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        # Define the online network architecture with LSTM layer
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            feed_tensor(),  # Custom module to reshape tensor before LSTM
            nn.LSTM(input_size=64 * 7 * 7, hidden_size=512, batch_first=True),
            extract_tensor(),  # Custom module to extract tensor from LSTM output
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        # Create a target network by deep copying the online network
        self.target = copy.deepcopy(self.online)

        # Freeze the parameters of the target network
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        """
        Forward pass through the DRDQN model.

        Parameters:
        - input (tensor): Input state tensor.
        - model (str): Specifies whether to use 'online' or 'target' network.

        Returns:
        - tensor: Output Q-values.
        """
        # Specify whether to use the online or target network during forward pass
        if model == 'online':
            return self.online(input)
        elif model == 'target':
            return self.target(input)

# Define a custom module to extract tensor from LSTM output
class extract_tensor(nn.Module):
    def forward(self, x):
        """
        Extract tensor from LSTM output.

        Parameters:
        - x (tuple): Tuple containing tensor and hidden states.

        Returns:
        - tensor: Extracted tensor.
        """
        tensor, _ = x
        return tensor.squeeze(1)

# Define a custom module to reshape tensor before feeding into LSTM
class feed_tensor(nn.Module):
    def forward(self, x):
        """
        Reshape tensor before feeding into LSTM.

        Parameters:
        - x (tensor): Input tensor.

        Returns:
        - tensor: Reshaped tensor.
        """
        x = x.view(x.size(0), -1)
        return x.unsqueeze(1)
