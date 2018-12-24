import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import exp, factorial
import seaborn as sns


MAX_CARS = 20
MAX_MOVE_OF_CAR = 5
MEAN_REQUESTS_A = 3
MEAN_REQUESTS_B = 4
MEAN_RETURN_A = 3
MEAN_RETURN_B = 2
DISCOUNT = 0.9
RENTAL_CREDIT = 10
MOVE_COST = 2

actions = np.arange(-MAX_MOVE_OF_CAR, MAX_MOVE_OF_CAR + 1)
poisson_cache = dict()
return_prob_cache = dict()


def poisson(n, lam):
  global poisson_cache
  try:
    value = poisson_cache[n * 10 + lam]
  except Exception:
    value = exp(-lam) * pow(lam, n) / factorial(n)
    poisson_cache[n * 10 + lam] = value
  return value


def return_prob_matrix(a, b):
  global return_prob_cache
  try:
    value = return_prob_cache[a * 30 + b]
  except Exception:
    value = np.zeros((MAX_CARS + 1, MAX_CARS + 1), dtype=np.float32)
    a_total_return_prob = 1.
    for a_return in range(MAX_CARS - a + 1):
      if a_return == MAX_CARS - a:
        a_return_prob = a_total_return_prob
      else:
        a_return_prob = poisson(a_return, MEAN_RETURN_A)
        a_total_return_prob -= a_return_prob
      b_total_return_prob = 1.
      for b_return in range(MAX_CARS - b + 1):
        if b_return == MAX_CARS - b:
          b_return_prob = b_total_return_prob
        else:
          b_return_prob = poisson(b_return, MEAN_RETURN_B)
          b_total_return_prob -= b_return_prob
        value[a + a_return, b + b_return] = a_return_prob * b_return_prob
    return_prob_cache[a * 30 + b] = value
  return value


def expected_return(state, action, state_value):
  G = 0.
  cars_a, cars_b = np.array(state) - np.array([action, -action])
  if np.min([cars_a, cars_b]) < 0:
    return -1000
  basic_cost = MOVE_COST * np.abs(action)
  G -= basic_cost
  a_total_rental_prob = 1.
  for a_rental_reqs in range(cars_a + 1):
    if a_rental_reqs == cars_a:
      a_prob = a_total_rental_prob
    else:
      a_prob = poisson(a_rental_reqs, MEAN_REQUESTS_A)
      a_total_rental_prob -= a_prob
    b_total_rental_prob = 1.
    for b_rental_reqs in range(cars_b + 1):
      if b_rental_reqs == cars_b:
        b_prob = b_total_rental_prob
      else:
        b_prob = poisson(b_rental_reqs, MEAN_REQUESTS_B)
        b_total_rental_prob -= b_prob
      joint_rental_prob = a_prob * b_prob
      basic_rental_credit = (a_rental_reqs + b_rental_reqs) * RENTAL_CREDIT
      G += joint_rental_prob * basic_rental_credit
      left_cars_a = cars_a - a_rental_reqs
      left_cars_b = cars_b - b_rental_reqs
      prob_matrix = return_prob_matrix(left_cars_a, left_cars_b)
      G += np.sum(joint_rental_prob * DISCOUNT * state_value * prob_matrix)
  return G


def figure_4_2():
  value = np.zeros((MAX_CARS + 1, MAX_CARS + 1), dtype=np.float32)
  policy = np.zeros(value.shape, dtype=np.int)

  iterations = 0
  _, axes = plt.subplots(2, 3, figsize=(40, 20))
  plt.subplots_adjust(wspace=0.1, hspace=0.2)
  axes = axes.flatten()
  while True:
    fig = sns.heatmap(np.flipud(policy), cmap="YlGnBu", ax=axes[iterations])
    fig.set_ylabel('# cars at first location', fontsize=30)
    fig.set_yticks(list(reversed(range(MAX_CARS + 1))))
    fig.set_xlabel('# cars at second location', fontsize=30)
    fig.set_title('policy %d' % (iterations), fontsize=30)

    while True:
      new_value = np.copy(value)
      for i in range(MAX_CARS + 1):
        for j in range(MAX_CARS + 1):
          new_value[i, j] = expected_return([i, j], policy[i, j], new_value)
      value_change = np.max(np.abs((new_value - value)))
      print('value change %f' % (value_change))
      value = new_value
      if value_change < 1e-4:
        break

    new_policy = np.copy(policy)
    for i in range(MAX_CARS + 1):
      for j in range(MAX_CARS + 1):
        action_returns = []
        for action in actions:
          action_returns.append(expected_return([i, j], action, value))
        new_policy[i, j] = actions[np.argmax(action_returns)]

    policy_change = (new_policy != policy).sum()
    print('policy changed in %d states' % (policy_change))
    policy = new_policy
    if policy_change == 0:
      fig = sns.heatmap(np.flipud(value), cmap="YlGnBu", ax=axes[-1])
      fig.set_ylabel('# cars at first location', fontsize=30)
      fig.set_yticks(list(reversed(range(MAX_CARS + 1))))
      fig.set_xlabel('# cars at second location', fontsize=30)
      fig.set_title('optimal value', fontsize=30)
      break

    iterations += 1

  plt.savefig('../images/figure_4_2.png')
  plt.close()


if __name__ == '__main__':
  figure_4_2()
