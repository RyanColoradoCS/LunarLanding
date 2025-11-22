from tkinter.font import names
import gymnasium as gym
import numpy as np
from collections import defaultdict


# Q-Learning Agent Outline for LunarLander-v3 with Discrete Actions
# I am using the Blackjack model as a template.
class LunarLanderAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.99,
    ):
        """
        Q-learning outline adapted from Blackjack agent.
        """

        self.env = env

        # Q-table
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        # Learning parameters
        self.lr = learning_rate
        self.discount_factor = discount_factor

        # Exploration
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

        # TODO: Create bins for each of the 8 dimensions in obs


    # TODO: discretize() — required for tabular Q-learning
    def discretize(self, obs: np.ndarray) -> tuple:
        """
        Convert continuous obs (8 floats) → discrete tuple.
        """
        raise NotImplementedError("You must implement discretization.")

    # get_action() — epsilon-greedy
    def get_action(self, obs: np.ndarray) -> int:
        """
        0: do nothing
        1: fire left orientation engine
        2: fire main engine
        3: fire right orientation engine
        
        LunarLander uses discrete actions 0–3.
        obs must be converted to a discrete state.
        """

        # TODO: convert obs → discrete state
        raise NotImplementedError("Fill in epsilon-greedy logic.")


    # update() — Q-learning Bellman update
    def update(self, obs, action, reward, terminated, truncated, next_obs):
        """
        Q-learning update:
        Q(s,a) ← Q(s,a) + α * (reward + γ * max_a' Q(s',a') − Q(s,a))
        """

        # TODO: discretize
        # state = self.discretize(obs)
        # next_state = self.discretize(next_obs)

        # TODO: handle done condition
        # done = terminated or truncated

        # TODO: compute future_q
        # if done:
        #     future_q = 0
        # else:
        #     future_q = np.max(self.q_values[next_state])

        # TODO: compute target
        # target = reward + self.discount_factor * future_q

        # TODO: TD error
        # td_error = target - self.q_values[state][action]

        # TODO: update Q-table
        # self.q_values[state][action] += self.lr * td_error

        # self.training_error.append(td_error)
        raise NotImplementedError("Implement Q-learning update.")

    # Epsilon decay
    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
        

def main():
    env = gym.make(
        "LunarLander-v3",
        continuous=False,
        render_mode="human",  # Set to "human" to visualize the environment
    )

    obs, info = env.reset(seed=42)
    print("Initial observation:", obs)
    print("Action space:", env.action_space)

    for step in range(1000):
        action = env.action_space.sample()  # random policy
        obs, reward, terminated, truncated, info = env.step(action)
        
        # environment outputs
        print(f"Step {step}")
        print("  obs:", obs)
        values = obs.tolist()
        names = ["x", "y", "vx", "vy", "angle", "angular velocity,", "leg1", "leg2"]
        for name, val in zip(names, values):
            print(f"{name}: {val}")
        print("  action:", action)
        print("  reward:", reward)
        print("  terminated:", terminated, " truncated:", truncated)
        print("-" * 40)
        
        values = obs.tolist()
        

        if terminated or truncated:
            print("Episode finished at step", step)
            obs, info = env.reset()

    env.close()

if __name__ == "__main__":
    main()
