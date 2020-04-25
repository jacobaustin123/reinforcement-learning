# general RL framework

# new position from (x, y) with action
def move(x, y, action):
    pass

# probability of each action
def policy(action):
    pass

# reward of moving from (x, y) with action
def reward(x, y, action):
    pass

policy:
    left: 0.5
    right: 0.5
    up: 0
    down: 0
move:
    left: (0, 1)
    all: "default"

reward:
    (3, 2) -> 5
    "any" -> "edge"

def run(policy, reward, move):
    pass