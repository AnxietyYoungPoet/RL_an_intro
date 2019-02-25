from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')

N_STATES = 19
STATES = np.arange(1, N_STATES + 1)
START_STATE = 10
END_STATES = [0, N_STATES + 1]
TRUE_VALUE = np.arange(-20, 22, 2) / 20.0
TRUE_VALUE[0] = TRUE_VALUE[N_STATES + 1] = 0.0


class Actions(object):
  def __init__(self):
    self.length = 5000000
    self._load_buffer()
    self.current = 0

  def _load_buffer(self):
    self.action_buffer = np.random.randint(2, size=self.length) * 2 - 1

  def get_action(self):
    if self.current + 1 > self.length:
      self.current = 0
      self._load_buffer()
    value = self.action_buffer[self.current]
    self.current += 1
    return value


def transition(state, action):
  new_state = state + action
  if new_state ==  END_STATES[0]:
    reward = -1
  elif new_state == END_STATES[1]:
    reward = 1
  else:
    reward = 0
  return reward, new_state


class ValueFunction(object):
  def __init__(self, rate, step_size):
    self.rate = rate
    self.step_size = step_size
    self.weights = np.zeros(N_STATES + 2)
  
  def value(self, state):
    return self.weights[state]
  
  def learn(self, state, reward):
    return
