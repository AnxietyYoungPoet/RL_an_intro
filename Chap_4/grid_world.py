#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.table import Table

WORLD_SIZE = 4
TERMINAL = ([0, 0], [3, 3])

# left, up, right, down
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]
ACTION_PROB = 0.25


def step(state, action):
  new_state = np.array(state) + action
  if np.max(new_state) > WORLD_SIZE - 0.5 or np.min(new_state) < -0.5:
    return state, -1
  return new_state.tolist(), -1


def draw_image(image):
  fig, ax = plt.subplots()
  ax.set_axis_off()
  tb = Table(ax, bbox=[0, 0, 1, 1])

  nrows, ncols = image.shape
  width, height = 1.0 / ncols, 1.0 / nrows

  # Add cells
  for (i, j), val in np.ndenumerate(image):
    # Index either the first or second item of bkg_colors based on
    # a checker board pattern
    # idx = [j % 2, (j + 1) % 2][i % 2]
    color = 'white'

    tb.add_cell(i, j, width, height, text=val, 
                loc='center', facecolor=color)

  # Row Labels...
  for i, label in enumerate(range(len(image))):
    tb.add_cell(i, -1, width, height, text=label + 1, loc='right', 
                edgecolor='none', facecolor='none')
  # Column Labels...
  for j, label in enumerate(range(len(image))):
    tb.add_cell(-1, j, width, height / 2, text=label + 1, loc='center', 
                edgecolor='none', facecolor='none')
  ax.add_table(tb)


def figure_4_1():
  value = np.zeros((WORLD_SIZE, WORLD_SIZE))
  while True:
    # keep iteration until convergence
    delta = 0
    for i in range(0, WORLD_SIZE):
      for j in range(0, WORLD_SIZE):
        if [i, j] in TERMINAL:
          continue
        old_value = value[i, j]
        temp_value = 0
        for action in ACTIONS:
          (next_i, next_j), reward = step([i, j], action)
          # bellman equation
          temp_value += ACTION_PROB * (reward + value[next_i, next_j])
        value[i, j] = temp_value
        if np.abs(old_value - temp_value) > delta:
          delta = np.abs(old_value - temp_value)
    if delta < 1e-5:
      draw_image(np.round(value, decimals=2))
      plt.savefig('../images/figure_4_1.png')
      plt.close()
      break


if __name__ == '__main__':
  figure_4_1()
  # figure_3_5()
