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
  
  def new_episode(self):
    return
  

class OffLineLambdaReturn(ValueFunction):
  def __init__(self, rate, step_size):
    ValueFunction.__init__(self, rate, step_size)
    self.rate_truncate = 1e-3
  
  def new_episode(self):
    self.trajectory = [START_STATE]
    self.reward = 0.0
  
  def learn(self, state, reward):
    self.trajectory.append(state)
    if state in END_STATES:
      self.reward = reward
      self.T = len(self.trajectory) - 1
      self.off_line_learn()
  
  def n_step_from_t(self, n, t):
    end_time = min(t + n, self. T)
    returns = self.value(self.trajectory[end_time])
    if end_time == self.T:
      returns += self.reward
    return returns
  
  def lambda_return_from_t(self, t):
    returns = 0.0
    lambda_power = 1.
    for n in range(1, self.T - t):
      returns += lambda_power * self.n_step_from_t(n, t)
      lambda_power *= self.rate
      if lambda_power < self.rate_truncate:
        break
      returns *= 1 - self.rate
      if lambda_power >= self.rate_truncate:
        returns += lambda_power * self.reward
    return returns
  
  def off_line_learn(self):
    for t in range(self.T):
      state = self.trajectory[t]
      delta = self.lambda_return_from_t(t) - self.value(state)
      delta *= self.step_size
      self.weights[state] += delta


class TemporalDifferenceLambda(ValueFunction):
  def __init__(self, rate, step_size):
    ValueFunction.__init__(self, rate, step_size)
    self.new_episode()

  def new_episode(self):
    self.eligibility = np.zeros(N_STATES + 2)
    self.last_state = START_STATE
  
  def learn(self, state, reward):
    self.eligibility *= self.rate
    self.eligibility[self.last_state] += 1
    delta = reward + self.value(state) - self.value(self.last_state)
    delta *= self.step_size
    self.weights += delta * self.eligibility
    self.last_state = state


class TrueOnlineTemporalDifferenceLambda(ValueFunction):
  def __init__(self, rate, step_size):
    ValueFunction.__init__(self, rate, step_size)

  def new_episode(self):
    # initialize the eligibility trace
    self.eligibility = np.zeros(N_STATES + 2)
    # initialize the beginning state
    self.last_state = START_STATE
    # initialize the old state value
    self.old_state_value = 0.0

  def learn(self, state, reward):
    # update the eligibility trace and weights
    last_state_value = self.value(self.last_state)
    state_value = self.value(state)
    dutch = 1 - self.step_size * self.rate * self.eligibility[self.last_state]
    self.eligibility *= self.rate
    self.eligibility[self.last_state] += dutch
    delta = reward + state_value - last_state_value
    self.weights += self.step_size * (
        delta + last_state_value - self.old_state_value) * self.eligibility
    self.weights[self.last_state] -= self.step_size * \
        (last_state_value - self.old_state_value)
    self.old_state_value = state_value
    self.last_state = state


def random_walk(value_function, actions):
  value_function.new_episode()
  state = START_STATE
  while state not in END_STATES:
    action = actions.get_action()
    reward, new_state = transition(state, action)
    value_function.learn(new_state, reward)
    state = new_state



def parameter_sweep(value_function_generator, runs, lambdas, alphas):
  # play for 10 episodes for each run
  episodes = 10
  actions = Actions()
  # track the rms errors
  errors = [np.zeros(len(alphas_)) for alphas_ in alphas]
  for run in tqdm(range(runs)):
    for lambdaIndex, rate in zip(range(len(lambdas)), lambdas):
      for alphaIndex, alpha in zip(range(len(alphas[lambdaIndex])), alphas[lambdaIndex]):
        valueFunction = value_function_generator(rate, alpha)
        for episode in range(episodes):
          random_walk(valueFunction, actions)
          stateValues = [valueFunction.value(state) for state in STATES]
          errors[lambdaIndex][alphaIndex] += np.sqrt(
              np.mean(np.power(stateValues - TRUE_VALUE[1: -1], 2)))

  # average over runs and episodes
  for error in errors:
    error /= episodes * runs
  for i in range(len(lambdas)):
    plt.plot(alphas[i], errors[i], label='lambda = ' + str(lambdas[i]))
  plt.xlabel('alpha')
  plt.ylabel('RMS error')
  plt.legend()


def figure_12_3():
  lambdas = [0.0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
  alphas = [np.arange(0, 1.1, 0.1),
            np.arange(0, 1.1, 0.1),
            np.arange(0, 1.1, 0.1),
            np.arange(0, 1.1, 0.1),
            np.arange(0, 1.1, 0.1),
            np.arange(0, 0.55, 0.05),
            np.arange(0, 0.22, 0.02),
            np.arange(0, 0.11, 0.01)]
  parameter_sweep(OffLineLambdaReturn, 50, lambdas, alphas)

  plt.savefig('../images/figure_12_3.png')
  plt.close()


def figure_12_6():
  lambdas = [0.0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
  alphas = [np.arange(0, 1.1, 0.1),
            np.arange(0, 1.1, 0.1),
            np.arange(0, 0.99, 0.09),
            np.arange(0, 0.55, 0.05),
            np.arange(0, 0.33, 0.03),
            np.arange(0, 0.22, 0.02),
            np.arange(0, 0.11, 0.01),
            np.arange(0, 0.044, 0.004)]
  parameter_sweep(TemporalDifferenceLambda, 50, lambdas, alphas)

  plt.savefig('../images/figure_12_6.png')
  plt.close()


def figure_12_8():
  lambdas = [0.0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
  alphas = [np.arange(0, 1.1, 0.1),
            np.arange(0, 1.1, 0.1),
            np.arange(0, 1.1, 0.1),
            np.arange(0, 1.1, 0.1),
            np.arange(0, 1.1, 0.1),
            np.arange(0, 0.88, 0.08),
            np.arange(0, 0.44, 0.04),
            np.arange(0, 0.11, 0.01)]
  parameter_sweep(TrueOnlineTemporalDifferenceLambda, 50, lambdas, alphas)

  plt.savefig('../images/figure_12_8.png')
  plt.close()


if __name__ == "__main__":
    # figure_12_3()
    # figure_12_6()
    figure_12_8()
