import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


VALUES = np.zeros(7)
VALUES[1:6] = 0.5
VALUES[6] = 1
TERMINALS = [0, 6]

TRUE_VALUE = np.zeros(7)
TRUE_VALUE[1:6] = np.arange(1, 6) / 6.0
TRUE_VALUE[6] = 1

ACTION_LEFT = 0
ACTION_RIGHT = 1


def TD(value, alpha=0.1):
  previous_state = 3
  state = previous_state
  trajectory = []
  while state not in TERMINALS:
    trajectory.append(state)
    state = state + 2 * np.random.binomial(1, 0.5) - 1
    value[previous_state] += alpha * (value[state] - value[previous_state])
    previous_state = state
  trajectory.append(state)
  return trajectory


def MC(value, alpha=0.1):
  state = 3
  trajectory = []
  while state not in TERMINALS:
    trajectory.append(state)
    state += 2 * np.random.binomial(1, 0.5) - 1
  if state == TERMINALS[-1]:
    reward = 1
  else:
    reward = 0
  for state_ in trajectory:
    value[state_] += alpha * (reward - value[state_])
  return reward, trajectory


def compute_state_value():
  plot_episodes = [0, 1, 10, 100]
  current_values = np.copy(VALUES)
  for i in range(plot_episodes[-1] + 1):
    if i in plot_episodes:
      plt.plot(current_values, label=str(i) + ' episodes')
    TD(current_values)
  plt.plot(TRUE_VALUE, label='true values')
  plt.xlabel('state')
  plt.ylabel('estimated value')
  plt.legend()


def rms_error():
  TD_alpha = [0.15, 0.1, 0.05]
  MC_alpha = [0.01, 0.02, 0.03, 0.04]
  episodes = 100
  runs = 100
  for alpha in TD_alpha:
    method = 'TD'
    linestyle = 'solid'
    errors = np.zeros(episodes + 1)
    for r in tqdm(range(runs), ncols=64):
      current_values = np.copy(VALUES)
      for i in range(episodes + 1):
        TD(current_values, alpha=alpha)
        errors[i] += np.sqrt(np.mean(np.square(current_values - TRUE_VALUE)))
    errors /= runs
    plt.plot(errors, linestyle=linestyle, label=method + ', alpha = %.02f' % (alpha))
  for alpha in MC_alpha:
    method = 'MC'
    linestyle = 'dashdot'
    errors = np.zeros(episodes + 1)
    for r in tqdm(range(runs), ncols=64):
      current_values = np.copy(VALUES)
      for i in range(episodes + 1):
        MC(current_values, alpha=alpha)
        errors[i] += np.sqrt(np.mean(np.square(current_values - TRUE_VALUE)))
    errors /= runs
    plt.plot(errors, linestyle=linestyle, label=method + ', alpha = %.02f' % (alpha))
  plt.xlabel('episodes')
  plt.ylabel('RMS')
  plt.legend()


def batch_updating(method):
  episodes = 100
  runs = 100
  errors = np.zeros(episodes + 1)
  alpha = 0.05
  for r in tqdm(range(runs)):
    history = []
    current_values = np.copy(VALUES)
    for i in range(episodes + 1):
      updates = np.zeros_like(current_values)
      if method == 'TD':
        history.append(TD(np.copy(current_values)))
        for trajectory in history:
          for index, state in enumerate(trajectory[:-1]):
            next_state = trajectory[index + 1]
            updates[state] += alpha * (current_values[next_state] - current_values[state])
        current_values += updates / len(history)
      else:
        history.append((MC(np.copy(current_values))))
        for reward, trajectory in history:
          for state in trajectory:
            updates[state] += alpha * (reward - current_values[state])
        current_values += updates / len(history)
      errors[i] += np.sqrt(np.mean(np.square(current_values - TRUE_VALUE)))
  return errors / runs


def example_6_2():
  plt.figure(figsize=(10, 20))
  plt.subplot(2, 1, 1)
  compute_state_value()

  plt.subplot(2, 1, 2)
  rms_error()
  plt.tight_layout()

  plt.savefig('../images/example_6_2.png')
  plt.close()


def figure_6_2():
  td_erros = batch_updating('TD')
  mc_erros = batch_updating('MC')

  plt.plot(td_erros, label='TD')
  plt.plot(mc_erros, label='MC')
  plt.xlabel('episodes')
  plt.ylabel('RMS error')
  plt.legend()

  plt.savefig('../images/figure_6_2.png')
  plt.close()


if __name__ == '__main__':
  # example_6_2()
  figure_6_2()
