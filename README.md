# LunarLanding
LunarLanding Project by Ryan Young

Brief Overview:
In this project, I use Deep Q-Learning to solve the Lunar Landing problem. I use a Deep Q Network (DQN) because regular Q-learning relies on a Q-table that stores the value of every possible state/action pair. This only works in small, discrete environments, and it quickly becomes impractical when the state space is large or continuous. In the LunarLander environment, the agent’s observation consists of eight continuous variables such as position, velocity, angle, and leg contact, which means there are an infinite number of possible states. Deep Q-Learning solves this issue by using a neural network to approximate the Q-function.

My solution: 
I had a simple approach to solving this problem. I had to use PyTorch to implement the neural networks. In this project, first I import the Open AI’s Gym for the environment, NumPy and random for data and sampling, PyTorch for implementing the neural network, and deque for the replay buffer.

Next, I defined my hyperparameters:
- The state size (8 observation values)
- Number of actions (4 discrete actions)
- The hidden layer size and learning rate
- The discount factor (gamma)
- The replay buffer size and batch size
- Training settings such as the number of episodes, max amount of steps, and how often to update the target network
- Epsilon for if the agent explores or exploits during training

As I mentioned above, the DQN class is a neural network that approximates the Q-values. It takes an 8 states as input and outputs 4 Q-values (one per possible action). It’s just a simple implementation that is implement as: Linear to ReLU to Linear to ReLU to Linear. ReLu stands for Rectified Linear Unit, which is an activation function in deep learning. PyTorch handles all the math and linear algebra.

I also created a ExperienceReplayBuffer Python class that holds the past experiences in a tuple that contains (state, action, reward, next_state, done) in a deque. It has an append function that pushes new transitions. There is also a function sample randomly that gets a batch and returns NumPy arrays for training.

LunarLandingDQNAgent is the agent that is being trained to solve the problem. The class consists of:
- An online network (the one trained every step)
- A target network (a periodically updated copy, which stabilizes training).
- The function store_experience that puts transitions into the replay buffer.
- The function update_policy that checks if there are enough samples and then calls train_from_batch.
- The function train_from_batch, where NumPy arrays are sampled from the buffer and are converted to PyTorch tensors.
- The select_action implements epsilon/greedy method of solving the problem. With probability eps, it picks a random action. Otherwise, it acts greedy, converting the state to a tensor, running it through online network, and choosing the argmax of Q.
In the LunarLandingDQNAgent class, I build the Bellman target using the equation target_q = r + gamma * max_next_q * (1 - done). Then, I calculate the MSE loss between current_q and target_q, and backprop with loss.backward() and optimizer.step().

The function train_dqn() creates the environment and trains the learning agent. For each episode, it resets the environment, then on each step, it picks an action with agent.select_action(state, eps), stores the transition, and then calls agent.update_policy() to train on a batch from experience replay memory. The function returns the trained agent and reward history and saves the online network’s weights.

Finally, my project then runs evaluate_agent function, which uses a greedy policy (since epsilon is set to 0.0) with render_mode="human" to visually display several episodes. It prints the per episode rewards and the average reward as the final evaluation. If it is an average of 200, then it is considered solved. With this approach, I was able to solve this problem.


My results:
If you can get an average score of 200, then you have solved the problem. With solution I explained above, I was able to consistently solve this problem.
Finding the right hyper parameters was difficult. I finally settled with the training agent always beginning with full exploration (always random actions). The exploration gradually decreases until it bottoms out at 5% random actions, so the agent still explores occasionally. The decay is eps_decay = 0.995. For each episode, epsilon is multiplied by 0.995, so it shrinks slowly toward eps_end (but never going lower). The following results are what I got from this method of choosing epsilon.
Output from my solution:
Eval Episode 1: Reward 278.2
Eval Episode 2: Reward 220.7
Eval Episode 3: Reward 304.3
Eval Episode 4: Reward -164.3
Eval Episode 5: Reward 305.8
Eval Episode 6: Reward 254.2
Eval Episode 7: Reward 101.4
Eval Episode 8: Reward 265.2
Eval Episode 9: Reward 238.5
Eval Episode 10: Reward 257.6
Average eval reward over 10 episodes: 206.2
Average reward over 100 episodes: 206.2

Reflections:

This problem was difficult to solve even though it is one of the easier problems on the site. First, I had to learn all about Deep Q-Learning. Honestly, my linear algebra was rusty, but with PyTorch I was able to handle all of that. The hard part was learning how to use it from the documentation and YouTuve videos. I also wish I had done a better job of documenting all the different hyper parameters I used. If I could go back, I would have used a Jupyter notebook and used matplotlib to document everything, but I had issues getting the notebook to work with Gym and PyTorch, and I had to get started on solving the assignment.

The biggest help to me in solving this with Deep Q-Learning was drawing everything out. Once I had a drawing of everything and how it would work, including the neural network and the Replay Buffer, the solution clicked. Then it was just about getting PyTorch to work and tuning the hyper parameters.

References:
The two biggest helps for me were these YouTube videos:
https://www.youtube.com/watch?v=EUrWGTCGzlA
https://www.youtube.com/watch?v=6kOvmZDEMdc&t=18s
I used these to learn how to solve this problem. The PyTorch documentation was also very helpful.
