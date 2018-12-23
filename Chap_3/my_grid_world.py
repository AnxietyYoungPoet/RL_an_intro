import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table


GRID_SIZE = 5
A_POS = [0, 1]
A_PRIME_POS = [4, 1]
B_POS = [0, 3]
B_PRIME_POS = [2, 3]
DISCOUNT = 0.9

ACTIONS = np.array(
  [0, -1], [0, 1], [-1, 0], [1, 0])


def next_state(state, action):
  if state == A_POS:
    return A_PRIME_POS, 10
  if state == B_POS:
    return B_PRIME_POS, 5
  new_state = np.array(state) + action
  if np.max(new_state) > GRID_SIZE - 0.5 or np.min(new_state) < -0.5:
    return state, -1
  return new_state.tolist(), 0
