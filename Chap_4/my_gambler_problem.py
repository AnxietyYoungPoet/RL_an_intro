import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 


GOAL = 100
HEAD_PROB = 0.4
STATES = np.arange(GOAL + 1)


def fig_4_3():
  values = np.zeros(STATES.shape)
  policy = np.zeros(values.shape, dtype=np.int)
  values[-1] = 1
  while True:
    delta = 0
    for i in range(len(values)):
      if i in [0, GOAL]:
        continue
      old_value = values[i]
      target_value = -1
      for action in range(1, min(i, GOAL - i) + 1):
        action_value = HEAD_PROB * values[i + action] + (1 - HEAD_PROB) * values[i - action]
        if action_value > target_value:
          target_value = action_value
      values[i] = target_value
      delta += np.abs(old_value - target_value)
    for state in STATES[1:GOAL]:
        actions = np.arange(min(state, GOAL - state) + 1)
        action_returns = []
        for action in actions:
            action_returns.append(
                HEAD_PROB * values[state + action] + (1 - HEAD_PROB) * values[state - action])
        policy[state] = actions[np.argmax(np.round(action_returns[1:], 5)) + 1]
    if delta < 1e-9:
      print(values)
      print(policy)
      break
  plt.figure(figsize=(10, 20))

  plt.subplot(2, 1, 1)
  plt.plot(values)
  plt.xlabel('Capital')
  plt.ylabel('Value estimates')

  plt.subplot(2, 1, 2)
  plt.scatter(STATES, policy)
  plt.xlabel('Capital')
  plt.ylabel('Final policy (stake)')

  plt.savefig('../images/figure_4_3.png')
  plt.close()


if __name__ == '__main__':
  fig_4_3()
