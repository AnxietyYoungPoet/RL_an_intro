import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

WORLD_HEIGHT = 4
WORLD_WIDTH = 12
EPSILON = 0.1
ALPHA = 0.5
ACTION_UP = np.array([-1, 0])
ACTION_DOWN = np.array([1, 0])
ACTION_LEFT = np.array([0, -1])
ACTION_RIGHT = np.array([0, 1])
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]
CLIFFS = [(3, i) for i in range(1, 11)]
START = (3, 0)
GOAL = (3, 11)


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
  next_x = np.clip(x + true_action[0], 0, WORLD_HEIGHT - 1)
  next_y = np.clip(y + true_action[1], 0, WORLD_WIDTH - 1)
  next_state = (next_x, next_y)
  if next_state in CLIFFS:
    reward = -100
  else:
    reward = -1
  return next_state, reward


def Q_learning(previous_state, action, reward, state, value):
  pre_x, pre_y = previous_state
  x, y = state
  if state in CLIFFS or state == GOAL:
    value[pre_x, pre_y, action] += ALPHA * (reward - value[pre_x, pre_y, action])
  else:
    value[pre_x, pre_y, action] += ALPHA * (
      reward + np.max(value[x, y]) - value[pre_x, pre_y, action])


def sarsa(previous_state, action, reward, state, value):
  pre_x, pre_y = previous_state
  x, y = state
  if state in CLIFFS or state == GOAL:
    value[pre_x, pre_y, action] += ALPHA * (reward - value[pre_x, pre_y, action])
    next_action = None
  else:
    next_action = policy(state, value)
    value[pre_x, pre_y, action] += ALPHA * (
      reward + value[x, y, next_action] - value[pre_x, pre_y, action])
  return next_action


def fig_6_4():
  episodes = 500
  runs = 100
  SARSA = np.zeros(episodes + 1)
  Q = np.zeros(episodes + 1)
  for r in tqdm(range(runs)):
    value = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
    for i in range(episodes + 1):
      previous_state = START
      state = previous_state
      while True:
        action = policy(state, value)
        state, reward = transition(state, action)
        Q_learning(previous_state, action, reward, state, value)
        Q[i] += reward
        if state in CLIFFS or state == GOAL:
          break
        previous_state = state
  Q /= runs
  for r in tqdm(range(runs)):
    value = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
    for i in range(episodes + 1):
      previous_state = START
      state = previous_state
      action = policy(state, value)
      while True:
        state, reward = transition(state, action)
        action = sarsa(previous_state, action, reward, state, value)
        SARSA[i] += reward
        if state in CLIFFS or state == GOAL:
          break
        previous_state = state
  SARSA /= runs
  plt.plot(SARSA, label='Sarsa')
  plt.plot(Q, label='Q-Learning')
  plt.xlabel('Episodes')
  plt.ylabel('Sum of rewards during episode')
  plt.ylim([-100, 0])
  plt.legend()

  plt.savefig('../images/figure_6_4.png')
  plt.close()


if __name__ == '__main__':
  fig_6_4()
