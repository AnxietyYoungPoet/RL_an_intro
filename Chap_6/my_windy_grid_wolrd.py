import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


WORLD_HEIGHT = 7

# world width
WORLD_WIDTH = 10

# wind strength for each column
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
ACTION_UP = np.array([-1, 0])
ACTION_DOWN = np.array([1, 0])
ACTION_LEFT = np.array([0, -1])
ACTION_RIGHT = np.array([0, 1])
EPSILON = 0.1
ALPHA = 0.5
REWARD = -1.0

START = (3, 0)
GOAL = (3, 7)
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]


def policy(state, value):
  x, y = state
  p = np.random.rand(1)[0]
  if p <= EPSILON:
    action = np.random.randint(0, 4)
  else:
    action = np.argmax(value[x, y])
  return action


def transition(state, action):
  true_action = ACTIONS[action]
  x, y = state
  wind = WIND[y]
  next_x = np.clip(x + true_action[0] - wind, 0, WORLD_HEIGHT - 1)
  next_y = np.clip(y + true_action[1], 0, WORLD_WIDTH - 1)
  reward = REWARD
  next_state = (next_x, next_y)
  return next_state, reward


def sarsa(previous_state, action, reward, state, value):
  pre_x, pre_y = previous_state
  x, y = state
  if state == GOAL:
    value[pre_x, pre_y, action] += ALPHA * (reward - value[pre_x, pre_y, action])
    next_action = None
  else:
    next_action = policy(state, value)
    value[pre_x, pre_y, action] += ALPHA * (
      reward + value[x, y, next_action] - value[pre_x, pre_y, action])
  return next_action


def fig_6_3():
  steps = 8000
  episodes = np.zeros(steps + 1)
  value = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
  previous_state = START
  state = previous_state
  action = policy(state, value)
  for step in range(steps + 1):
    state, reward = transition(state, action)
    action = sarsa(previous_state, action, reward, state, value)
    if state == GOAL:
      episodes[step] = episodes[step - 1] + 1
      state = START
      action = policy(state, value)
    else:
      if step != 0:
        episodes[step] = episodes[step - 1]
    previous_state = state
  state = START
  while state != GOAL:
    print(state)
    action = np.argmax(value[state[0], state[1]])
    state, _ = transition(state, action)

  plt.plot(episodes)
  plt.xlabel('Time steps')
  plt.ylabel('Episodes')

  plt.savefig('../images/figure_6_3.png')
  plt.close()


if __name__ == '__main__':
  fig_6_3()
