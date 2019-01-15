import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


N_STATES = 1000
STATES = np.arange(1, N_STATES + 1)
START_STATE = 500
END_STATES = [0, N_STATES + 1]

ACTION_LEFT = -1
ACTION_RIGHT = 1
ACTIONS = [ACTION_LEFT, ACTION_RIGHT]

STEP_RANGE = 100


def compute_true_value():
  # true state value, just a promising guess
  true_value = np.arange(-1001, 1003, 2) / 1001.0

  # Dynamic programming to find the true state values, based on the promising guess above
  # Assume all rewards are 0, given that we have already given value -1 and 1 to terminal states
  while True:
    old_value = np.copy(true_value)
    for state in STATES:
      true_value[state] = 0
      for action in ACTIONS:
        for step in range(1, STEP_RANGE + 1):
          step *= action
          next_state = state + step
          next_state = max(min(next_state, N_STATES + 1), 0)
          # asynchronous update for faster convergence
          true_value[state] += 1.0 / (2 * STEP_RANGE) * true_value[next_state]
    error = np.sum(np.abs(old_value - true_value))
    if error < 1e-2:
      break
  # correct the state value for terminal states to 0
  true_value[0] = true_value[-1] = 0

  return true_value


class Actions(object):
  def __init__(self):
    self.length = 5000000
    self._load_buffer()
    self.current = 0

  def _load_buffer(self):
    self.action_buffer = np.random.randint(1, STEP_RANGE + 1, self.length)
    self.direction_buffer = np.random.binomial(1, 0.5, size=self.length) * 2 - 1

  def get_action(self):
    if self.current + 1 > self.length:
      self.current = 0
      self._load_buffer()
    value = self.action_buffer[self.current] * self.direction_buffer[self.current]
    self.current += 1
    return value


def transition(state, action):
  new_state = np.clip(state + action, END_STATES[0], END_STATES[1])
  if new_state == END_STATES[0]:
    reward = -1
  elif new_state == END_STATES[1]:
    reward = 1
  else:
    reward = 0
  return reward, new_state


def gradient_Monte_Carlo(value, actions, distribution, alpha=0.1):
  state = START_STATE
  trajectory = []
  while state not in END_STATES:
    trajectory.append(state)
    reward, state = transition(state, actions.get_action())
  for state in trajectory:
    group_state = (state - 1) // STEP_RANGE
    distribution[state] += 1
    value[group_state] += alpha * (reward - value[group_state])


def fig_9_1(true_value):
  episodes = int(1e5)
  alpha = 2e-5

  actions = Actions()
  group_value = np.zeros(N_STATES // STEP_RANGE)
  distribution = np.zeros(N_STATES + 1)
  for ep in tqdm(range(episodes)):
    gradient_Monte_Carlo(group_value, actions, distribution, alpha)
  distribution /= np.sum(distribution)
  state_values = [group_value[i // STEP_RANGE] for i in range(N_STATES)]

  plt.figure(figsize=(10, 20))

  plt.subplot(2, 1, 1)
  plt.plot(STATES, state_values, label='Approximate MC value')
  plt.plot(STATES, true_value[1: -1], label='True value')
  plt.xlabel('State')
  plt.ylabel('Value')
  plt.legend()

  plt.subplot(2, 1, 2)
  plt.plot(STATES, distribution[1:], label='State distribution')
  plt.xlabel('State')
  plt.ylabel('Distribution')
  plt.legend()

  plt.savefig('../images/figure_9_1.png')
  plt.close()


if __name__ == '__main__':
  true_value = compute_true_value()
  fig_9_1(true_value)
