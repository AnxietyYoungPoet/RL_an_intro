import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

N_STATES = 19

# discount
GAMMA = 1

# all states but terminal states
STATES = np.arange(1, N_STATES + 1)

# start from the middle state
START_STATE = 10

# two terminal states
# an action leading to the left terminal state has reward -1
# an action leading to the right terminal state has reward 1
END_STATES = [0, N_STATES + 1]

# true state value from bellman equation
TRUE_VALUE = np.arange(-20, 22, 2) / 20.0
TRUE_VALUE[0] = TRUE_VALUE[-1] = 0


class Actions(object):
  def __init__(self):
    self.length = 5000000
    self._load_buffer()
    self.current = 0

  def _load_buffer(self):
    self.action_buffer = np.random.randint(0, 2, self.length) * 2 - 1

  def get_action(self):
    if self.current + 1 > self.length:
      self.current = 0
      self._load_buffer()
    value = self.action_buffer[self.current]
    self.current += 1
    return value


def transition(state, action):
  new_state = state + action
  if new_state == END_STATES[0]:
    reward = -1
  elif new_state == END_STATES[1]:
    reward = 1 
  else:
    reward = 0
  return new_state, reward


def n_step_TD(n, alpha, value, actions):
  T = np.float('inf')
  t = 0
  reward_sequence = np.zeros(n)
  gamma_sequence = np.power(GAMMA, np.arange(0, n))
  state_sequence = [START_STATE for i in range(n)]
  while True:
    if t < T:
      state, reward = transition(state_sequence[-1], actions.get_action())
      reward_sequence[0: -1] = reward_sequence[1:]
      reward_sequence[-1] = reward
      if state in END_STATES:
        T = t + 1
    else:
      reward_sequence[0: -1] = reward_sequence[1:]
      reward_sequence[-1] = 0
    tau = t - n + 1
    if tau >= 0:
      G = np.sum(reward_sequence * gamma_sequence)
      if tau + n < T:
        G += value[state]
      value[state_sequence[0]] += alpha * (G - value[state_sequence[0]])
    state_sequence[0: -1] = state_sequence[1:]
    state_sequence[-1] = state
    if tau == (T - 1):
      break
    t += 1
  

def fig_7_1():
  actions = Actions()
  ns = np.power(2, np.arange(10))
  periods = 10
  runs = 100
  alphas = np.arange(0, 1.1, 0.1)
  RMS = np.zeros((len(ns), len(alphas)))
  for r in tqdm(range(runs), ncols=64):
    for i, n in enumerate(ns):
      for j, alpha in enumerate(alphas):
        value = np.zeros(N_STATES + 2)
        for p in range(periods):
          n_step_TD(n, alpha, value, actions) 
          RMS[i, j] += np.sqrt(np.mean(np.square(value - TRUE_VALUE)))
  RMS /= (runs * periods)
  for i in range(0, len(ns)):
    plt.plot(alphas, RMS[i, :], label='n = %d' % (ns[i]))
  plt.xlabel('alpha')
  plt.ylabel('RMS error')
  plt.ylim([0.25, 0.55])
  plt.legend()

  plt.savefig('../images/figure_7_2.png')
  plt.close()


if __name__ == '__main__':
  fig_7_1()
