import gymnasium as gym
import numpy as np
from collections import defaultdict
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# pytorch model file path
MODEL_PATH = "lunar_agent.pt"

# Hyperparameters
obs_state_values = 8 # 8 values from observation space
num_actions = 4 # 4 discrete actions
hidden_layer_size = 64 # size of hidden layers
learning_rate = 1e-3
gamma = 0.99  # discount factor
replay_memory_capacity = 10000
replay_sample_size = 64  # number of experiences sampled per update

# Training settings
num_episodes = 1000
max_steps = 1000
target_update_freq = 10

# Environment name for Gym
env_name = "LunarLander-v3"

# Epsilon settings
eps_start = 1.0
eps_end = 0.05
eps_decay = 0.995

# DQN Network - define the model architecture
class DQN(nn.Module):
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),  # Linear layer from state_size to hidden_size
            nn.ReLU(), # Rectified Linear Unit activation function
            nn.Linear(hidden_size, hidden_size), # Another hidden layer
            nn.ReLU(), # Another ReLU activation
            nn.Linear(hidden_size, action_size), # Output Q-values for each of the 4 actions
        )

    def forward(self, x):
        return self.net(x)

# Experience Replay Buffer - define memory for experence replay
class ExperienceReplayBuffer:

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    # Appends a new experience into the deque/buffer. Each experience is a tuple of
    # (state, action, reward, next_state, terminated)
    def append(self, state, action, reward, next_state, terminated):
        self.buffer.append((state, action, reward, next_state, terminated))

    # Randomly samples a batch of experiences from the buffer
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, terminated = zip(*batch)

        # Return the batched experiences as numpy arrays
        return (
            np.vstack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.vstack(next_states),
            np.array(terminated, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)

# DQN Agent for Lunar Lander - This class consists of a dqn and a relay buffer
# This is an agent that can learn to play Lunar Lander.
class LunarLandingDQNAgent:
    def __init__(self):

        # Discount factor for long-term rewards
        self.discount = gamma

        # Action space size alias
        self.actions = num_actions

        # Batch size for training
        self.replay_sample_size = replay_sample_size
        
        # Initialize learning networks
        self.online_net = DQN(obs_state_values, self.actions, hidden_layer_size)
        self.target_net = DQN(obs_state_values, self.actions, hidden_layer_size)

        # Initialize target network weights
        self._initialize_target_net()

        # Optimizer setup
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Experience replay storage
        self.memory = ExperienceReplayBuffer(replay_memory_capacity)

    # Copy the weights into the target network
    def _initialize_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

    # Insert new (s, a, r, s', done) tuple into memory
    def store_experience(self, s, a, r, s_next, is_terminal):
        self.memory.append(s, a, r, s_next, is_terminal)

    # Choose action using epsilon/greedy exploration
    def select_action(self, state, eps: float = 0.0) -> int:

        # Random move (exploration)
        if random.random() < eps:
            return random.randrange(self.actions)

        # Greedy move (exploitation)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        # Get Q-values from online network
        with torch.no_grad():
            q_values = self.online_net(state_tensor)
        
        # Return the action with the highest Q-value
        return int(q_values.argmax(dim=1))

    # Update policy by sampling from experience replay
    def update_policy(self):
        if len(self.memory) < self.replay_sample_size:
            return  # not enough data means no training yet

        batch = self.memory.sample(self.replay_sample_size)
        self.train_from_batch(*batch)

    # Core gradient update from a batch of experiences
    def train_from_batch(self, states, actions, rewards, next_states, dones):

        # Tensor conversions
        s  = torch.tensor(states, dtype=torch.float32)
        ns = torch.tensor(next_states, dtype=torch.float32)
        a  = torch.tensor(actions, dtype=torch.long)
        r  = torch.tensor(rewards, dtype=torch.float32)
        d  = torch.tensor(dones, dtype=torch.float32)

        # Q(s,a) — values the online network predicted for chosen actions
        current_q = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Q_target(s', max_a')
        with torch.no_grad():
            max_next_q = self.target_net(ns).max(dim=1)[0]
            target_q = r + self.discount * max_next_q * (1.0 - d)

        # Loss and backward pass
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # Update target network from online network
    def update_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())


# This function trains the DQN agent in the Lunar Lander environment
def train_dqn():
    
    # Create the environment
    env = gym.make("LunarLander-v3")
    # Create DQN agent
    agent = LunarLandingDQNAgent()

    # Epsilon for exploration
    eps = eps_start
    # List to store rewards history
    rewards_history = []

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0.0

        for t in range(max_steps):
            # Perform one action in the environment
            action = agent.select_action(state, eps)
            next_state, reward, terminated, truncated, info = env.step(action)

            # Store experience and update policy
            agent.store_experience(state, action, reward, next_state, (terminated or truncated))
            agent.update_policy()
            state = next_state
            episode_reward += reward

            # Check for episode termination
            if terminated or truncated:
                break

        # Epsilon decay
        eps = max(eps_end, eps * eps_decay)

        # Periodically update target network
        if (episode + 1) % target_update_freq == 0:
            agent.update_target_network()

        # Store episode reward
        rewards_history.append(episode_reward)
        
        print(
            f"Episode {episode+1}/{num_episodes} | "
            f"Reward: {episode_reward:.1f} | Eps: {eps:.3f}"
        )

    # Close the environment
    env.close()
    # Return the trained agent and rewards history
    return agent, rewards_history


# Evaluate the trained agent and print detailed step information
def evaluate_agent(env_name, agent, episodes=10):
    # Create the environment
    env = gym.make("LunarLander-v3", render_mode="human", continuous=False)
    total_reward = 0.0

    for ep in range(episodes):
        state, info = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            
            action = agent.select_action(state, eps=0.0)  # greedy action selection
            next_state, reward, terminated, truncated, info = env.step(action) # take action in env
            
            ''' Print detailed information about the environment and action taken
            print("Observational space shape", env.observation_space.shape)
            print("Sample obervation:", env.observation_space.sample())

            print("Action space n", env.action_space.n)
            print("Sample action:", env.action_space.sample())
        

            # 0: do nothing
            # 1: fire left orientation engine
            # 2: fire main engine
            # 3: fire right orientation engine
            # LunarLander uses discrete actions 0–3.

            # Do this action in the environment
            # Note: truncated: The episode ended because of an artificial cutoff, 
            # not because of the environment’s natural rules
            
            # environment outputs
            print("Action taken:", action)
            print(f"Step {ep}")
            print("obs:", next_state)
            values = next_state.tolist()
            names = ["Horizontal Pad coordinate (x)", "Vertical pad coordinate (y)", "vx", "vy", "angle", "angular speed,", "leg1", "leg2"]
            for name, val in zip(names, values):
                print(f"{name}: {val}")
            print("Reward:", reward)
            print("Terminated:", terminated, " truncated:", truncated)
            print("Info:", info)
            print("-" * 40)
            '''
                        
            # Update state and accumulate reward
            done = terminated or truncated
            state = next_state
            ep_reward += reward

        print(f"Eval Episode {ep+1}: Reward {ep_reward:.1f}")
        total_reward += ep_reward

    env.close()
    avg_reward = total_reward / episodes
    print(f"Average eval reward over {episodes} episodes: {avg_reward:.1f}")
    return avg_reward


if __name__ == "__main__":
    
    # If we already trained the agent earlier in this session, re-use it
    if os.path.exists(MODEL_PATH):
        print("Loading agent from file...")
        agent = LunarLandingDQNAgent()
        agent.online_net.load_state_dict(torch.load(MODEL_PATH))
        agent.update_target_network()
        trained_agent = agent
    else:
        print("Training agent...this may take a while.")
        trained_agent, rewards_history = train_dqn()
        torch.save(trained_agent.online_net.state_dict(), MODEL_PATH)
        print("Training complete!")

    # Evaluate 10 episodes
    # NOTE:From the documentation, an episode is considered a solution if it scores at least 200 points.
    avg_reward = evaluate_agent(env_name, trained_agent, episodes=20)
    print(f"Average reward over 20 episodes: {avg_reward:.1f}")
