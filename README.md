# reinforcement-learning
Demonstrations, examples, and tutorials for reinforcement learning, including policy gradient and k-armed bandit problems.

## k-armed-bandit.py

`k-armed-bandit.py` compares the average reward for epsilon-greedy k-armed bandit algorithms over some number of iterations. These algorithms included the classic epsilon-greedy algorithm, the UCB algorithm, and the gradient bandits algorithm.

![epsilon-greedy.png](https://raw.githubusercontent.com/ja3067/reinforcement-learning/master/epsilon-greedy.png)

## breakout.py

These scripts play the Atari game Breakout in the OpenAI Gym environment. To use, `pip install gym` and `pip install gym[Atari]`, then run the script. The manual version uses a simple optimal control scheme, while the breakout.py script uses reinforcement Q-learning.

## inverted-pendulum.py

These scripts solved the optimal control problem using reinforcement Q-learning in the OpenAI Gym environment. To use, `pip install gym`, then run the script. The manual version uses a simple optimal control scheme instead of reinforcement learning.
