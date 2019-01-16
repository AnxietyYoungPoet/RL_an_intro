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


class Value(object):
  def __init__(self, group_size=100):
    self.group_size = group_size
    self.group_num = N_STATES // self.group_size
    self.get_params()

  def get_params(self):
    self.params = np.zeros(self.group_num)

  def value(self, state):
    if state in END_STATES:
      return 0
    group_index = (state - 1) // self.group_size
    return self.params[group_index]

  def update(self, delta, state):
    group_index = (state - 1) // self.group_size
    self.params[group_index] += delta


class BasisValue(object):
  def __init__(self, method, order):
    self.method = method
    self.order = order
    self.get_params()

  def get_params(self):
    self.params = np.zeros(self.order + 1)

  def get_features(self, state):
    state /= float(N_STATES)
    features = np.zeros(self.order + 1)
    if self.method == 'poly':
      for i in range(self.order + 1):
        features[i] = np.power(state, i)
    else:
      for i in range(self.order + 1):
        features[i] = np.cos(i * np.pi * state)
    return features

  def value(self, state):
    features = self.get_features(state)
    return np.dot(self.params, features)

  def update(self, delta, state):
    gradient = self.get_features(state)
    self.params += delta * gradient


def transition(state, action):
  new_state = np.clip(state + action, END_STATES[0], END_STATES[1])
  if new_state == END_STATES[0]:
    reward = -1
  elif new_state == END_STATES[1]:
    reward = 1
  else:
    reward = 0
  return reward, new_state


def gradient_Monte_Carlo(value, actions, distribution=None, alpha=0.1):
  state = START_STATE
  trajectory = []
  while state not in END_STATES:
    trajectory.append(state)
    reward, state = transition(state, actions.get_action())
  for state in trajectory:
    if distribution is not None:
      distribution[state] += 1
    delta = alpha * (reward - value.value(state))
    value.update(delta, state)


def gradient_TD(value, actions, alpha=0.1, n=1, group_size=100):
  T = np.float('inf')
  t = 0
  reward_sequence = np.zeros(n)
  state_sequence = [START_STATE for i in range(n)]
  while True:
    if t < T:
      reward, state = transition(state_sequence[-1], actions.get_action())
      reward_sequence[:-1] = reward_sequence[1:]
      reward_sequence[-1] = reward
      if state in END_STATES:
        T = t + 1
    else:
      reward_sequence[:-1] = reward_sequence[1:]
      reward_sequence[-1] = 0
    tau = t - n + 1
    if tau >= 0:
      G = np.sum(reward_sequence)
      if tau + n < T:
        G += value.value(state)
      delta = alpha * (G - value.value(state_sequence[0]))
      value.update(delta, state_sequence[0])
    state_sequence[:-1] = state_sequence[1:]
    state_sequence[-1] = state
    if tau == (T - 1):
      break
    t += 1


def fig_9_1(true_value):
  episodes = int(1e5)
  alpha = 2e-5

  actions = Actions()
  value = Value(100)
  # group_value = np.zeros(N_STATES // STEP_RANGE)
  distribution = np.zeros(N_STATES + 1)
  for ep in tqdm(range(episodes)):
    gradient_Monte_Carlo(value, actions, distribution, alpha)
  distribution /= np.sum(distribution)
  state_values = [value.value(i) for i in range(1, N_STATES + 1)]

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


def fig_9_2_left(true_value):
  episodes = int(1e5)
  alpha = 2e-4

  actions = Actions()
  value = Value()
  for ep in tqdm(range(episodes)):
    gradient_TD(value, actions, alpha=alpha)
  state_values = [value.value(i) for i in range(1, N_STATES + 1)]
  plt.plot(STATES, state_values, label='Approximate TD value')
  plt.plot(STATES, true_value[1: -1], label='True value')
  plt.xlabel('State')
  plt.ylabel('Value')
  plt.legend()


def fig_9_2_right(true_value):
  actions = Actions()
  ns = np.power(2, np.arange(10))
  periods = 10
  runs = 100
  alphas = np.arange(0, 1.1, 0.1)
  RMS = np.zeros((len(ns), len(alphas)))
  for r in tqdm(range(runs), ncols=64):
    for i, n in enumerate(ns):
      for j, alpha in enumerate(alphas):
        value = Value(50)
        for p in range(periods):
          gradient_TD(value, actions, alpha=alpha, n=n, group_size=50) 
          state_values = np.asarray([value.value(i) for i in range(1, N_STATES + 1)])
          RMS[i, j] += np.sqrt(np.mean(np.square(state_values - true_value[1: -1])))
  RMS /= (runs * periods)
  for i in range(0, len(ns)):
    plt.plot(alphas, RMS[i, :], label='n = %d' % (ns[i]))
  plt.xlabel('alpha')
  plt.ylabel('RMS error')
  plt.ylim([0.25, 0.55])
  plt.legend()


def fig_9_2(true_value):
  plt.figure(figsize=(10, 20))
  plt.subplot(2, 1, 1)
  fig_9_2_left(true_value)
  plt.subplot(2, 1, 2)
  fig_9_2_right(true_value)

  plt.savefig('../images/figure_9_2.png')
  plt.close()


def fig_9_5(true_value):
  runs = 1

  episodes = 5000

  # # of bases
  orders = [5, 10, 20]

  alphas = [1e-4, 5e-5]
  labels = [['polynomial basis'] * 3, ['fourier basis'] * 3]

  # track errors for each episode
  actions = Actions()
  errors = np.zeros((len(alphas), len(orders), episodes))
  for run in range(runs):
    for i in range(len(orders)):
      value_functions = [BasisValue('poly', orders[i]), BasisValue('fourier', orders[i])]
      for j in range(len(value_functions)):
        for episode in tqdm(range(episodes)):

          # gradient Monte Carlo algorithm
          gradient_Monte_Carlo(value_functions[j], actions, alpha=alphas[j])

          # get state values under current value function
          state_values = [value_functions[j].value(state) for state in STATES]

          # get the root-mean-squared error
          errors[j, i, episode] += np.sqrt(np.mean(np.power(true_value[1: -1] - state_values, 2)))

  # average over independent runs
  errors /= runs

  for i in range(len(alphas)):
    for j in range(len(orders)):
      plt.plot(errors[i, j, :], label='%s order = %d' % (labels[i][j], orders[j]))
  plt.xlabel('Episodes')
  plt.ylabel('RMSVE')
  plt.legend()

  plt.savefig('../images/figure_9_5.png')
  plt.close()


if __name__ == '__main__':
  true_value = compute_true_value()
  # fig_9_1(true_value)
  # fig_9_2(true_value)
  fig_9_5(true_value)
