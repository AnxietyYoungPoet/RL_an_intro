import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


ALPHA = 0.1
GAMMA = 0.95


class Maze(object):
  def __init__(self, obstacles=None, start_state=None):
    self.height = 6
    self.width = 9
    if start_state is None:
      self.start_state = (3, 1)
    else:
      self.start_state = start_state
    self.goal_state = (1, 9)
    if obstacles is None:
      self.obstacles = [[5, 6]]
      self.obstacles.extend([[i, 3] for i in range(2, 5)])
      self.obstacles.extend([[i, 8] for i in range(1, 4)])
    else:
      self.obstacles = obstacles
    self._init_maze()
    self._set_obstacles()
    self._load_action()
    # print(self._maze)

  def _load_action(self):
    up = [-1, 0]
    down = [1, 0]
    left = [0, -1]
    right = [0, 1]
    self.actions = [up, down, left, right]

  def _init_maze(self):
    self._maze = np.zeros((self.height + 2, self.width + 2))
    for i in range(self.width + 2):
      self._maze[0, i] = -1
      self._maze[self.height + 1, i] = -1
    for i in range(self.height + 2):
      self._maze[i, 0] = -1
      self._maze[i, self.width + 1] = -1
    self._maze[self.goal_state[0], self.goal_state[1]] = 1

  def _set_obstacles(self):
    for obstacle in self.obstacles:
      self._maze[obstacle[0], obstacle[1]] = -1

  def transition(self, state, act):
    action = self.actions[act]
    x, y = state
    new_x, new_y = x + action[0], y + action[1]
    if self._maze[new_x, new_y] == -1:
      return state, 0
    new_state = (new_x, new_y)
    if self._maze[new_x, new_y] == 1:
      return new_state, 1
    return new_state, 0


class Maze_model(object):
  def __init__(self, k=0):
    self.model = {}
    self.k = k
    self.time = 0
    self.actions = list(range(4))

  def init_model(self, k=0):
    self.model = {}
    self.time = 0
    self.k = k

  def record(self, state_action, state, reward):
    self.time += 1
    pre_state, action = state_action
    if bool(self.k):
      flag = True
      for action_ in self.actions:
        if (pre_state, action_) in self.model.keys():
          flag = False
          break
      if flag:
        for action_ in self.actions:
          if action_ != action:
            self.model[(pre_state, action_)] = (state, 0, 1)
    self.model[state_action] = (state, reward, self.time)

  def sample(self):
    (state_0, action_0), (state_, reward_, t) = random.choice(list(self.model.items()))
    if bool(self.k):
      reward_ += self.k * np.sqrt(self.time - t)
    return (state_0, action_0), (state_, reward_)


class Epsilon_greedy(object):
  def __init__(self, epsilon=0.1, seed=47):
    self.length = 100000
    self.rng = np.random.RandomState(seed)
    self._load_epsilon_buffer()
    self._load_action_buffer()
    self.p_eps = 0
    self.p_action = 0
    self.epsilon = epsilon

  def _load_epsilon_buffer(self):
    self.epsilon_buffer = self.rng.rand(self.length)

  def _load_action_buffer(self):
    self.action_buffer = self.rng.randint(0, 4, self.length)

  def get_action(self):
    if self.p_action + 1 > self.length:
      self.p_action = 0
      self._load_action_buffer()
    res = self.action_buffer[self.p_action]
    self.p_action += 1
    return res

  def get_epsilon(self):
    if self.p_eps + 1 > self.length:
      self.p_eps = 0
      self._load_epsilon_buffer()
    res = self.epsilon_buffer[self.p_eps]
    self.p_eps += 1
    return res

  def soft_action(self, values):
    epsilon = self.get_epsilon()
    if epsilon < self.epsilon:
      action_index = self.get_action()
    else:
      action_index = self.rng.choice(
        [action for action, value in enumerate(values) if value == np.max(values)])
    return action_index


def vanila_Q(value, pre_state, action, reward, state):
  pre_x, pre_y = pre_state
  x, y = state
  value[pre_x, pre_y, action] += ALPHA * (
    reward + GAMMA * np.max(value[x, y]) - value[pre_x, pre_y, action])


def dyna_Q(n, maze, ep_greedy, value, maze_model):
  pre_state = maze.start_state
  state = pre_state
  step = 0
  while state != maze.goal_state:
    action = ep_greedy.soft_action(value[state[0], state[1]])
    state_action = (pre_state, action)
    state, reward = maze.transition(pre_state, action)
    vanila_Q(value, pre_state, action, reward, state)
    maze_model.record(state_action, state, reward)
    pre_state = state
    for i in range(n):
      (state_0, action_0), (state_, reward_) = maze_model.sample()
      vanila_Q(value, state_0, action_0, reward_, state_)
    step += 1
  return step


def dyna_Q_plus(n, maze, ep_greedy, value, maze_model):
  pre_state = maze.start_state
  state = pre_state
  step = 0
  while state != maze.goal_state:
    action = ep_greedy.soft_action(value[state[0], state[1]])
    state_action = (pre_state, action)
    state, reward = maze.transition(pre_state, action)
    vanila_Q(value, pre_state, action, reward, state)
    maze_model.record(state_action, state, reward)
    pre_state = state
    for i in range(n):
      (state_0, action_0), (state_, reward_) = maze_model.sample()
      vanila_Q(value, state_0, action_0, reward_, state_)
    step += 1
  return step


def fig_8_2():
  ns = [0, 5, 50]
  runs = 30
  episodes = 50
  steps = np.zeros((len(ns), episodes))
  maze = Maze()
  maze_model = Maze_model()
  for r in tqdm(range(runs), ncols=64):
    seed = np.random.randint(1, 10000)
    for index, n in enumerate(ns):
      maze_model.init_model()
      ep_greedy = Epsilon_greedy(seed=seed)
      value = np.zeros((maze.height + 2, maze.width + 2, 4))
      for ep in range(episodes):
        step = dyna_Q(n, maze, ep_greedy, value, maze_model)
        steps[index, ep] += step
  steps /= runs
  for i in range(len(ns)):
    plt.plot(steps[i, :], label='%d planning steps' % (ns[i]))
  plt.xlabel('episodes')
  plt.ylabel('steps per episode')
  plt.legend()

  plt.savefig('../images/figure_8_2.png')
  plt.close()


def fig_8_4():
  global ALPHA
  ALPHA = 1.0
  k = 1e-4
  obstacles = [[4, i] for i in range(1, 9)]
  new_obstacles = [[4, i] for i in range(2, 10)]
  start_state = (6, 4)
  maze_model = Maze_model()
  change_step = 1000
  n = 10
  runs = 20
  max_step = 3000
  cum_reward = np.zeros((runs, 2, max_step))
  for r in tqdm(range(runs), ncols=64):
    seed = np.random.randint(1, 10000)
    for q in range(2):
      change_flag = True
      if q == 1:
        maze_model.init_model(k)
      else:
        maze_model.init_model()
      maze = Maze(obstacles=obstacles, start_state=start_state)
      step = 0
      last_step = step
      reward = 0
      value = np.zeros((maze.height + 2, maze.width + 2, 4))
      ep_greedy = Epsilon_greedy(seed=seed)
      while step < max_step:
        if q == 0:
          step += dyna_Q(n, maze, ep_greedy, value, maze_model)
        else:
          step += dyna_Q_plus(n, maze, ep_greedy, value, maze_model)
        cum_reward[r, q, last_step: min(step, max_step)] = reward
        reward += 1
        last_step = step
        if change_flag and step > change_step:
          maze = Maze(obstacles=new_obstacles, start_state=start_state)
          change_flag = False
  cum_reward = cum_reward.mean(axis=0)
  plt.plot(cum_reward[0], label='dyna-Q')
  plt.plot(cum_reward[1], label='dyna-Q+')
  plt.xlabel('time steps')
  plt.ylabel('cumulative reward')
  plt.legend()

  plt.savefig('../images/figure_8_4.png')
  plt.close()


def fig_8_5():
  global ALPHA
  ALPHA = 1.0
  k = 1e-3
  obstacles = [[4, i] for i in range(2, 10)]
  new_obstacles = [[4, i] for i in range(2, 9)]
  start_state = (6, 4)
  maze_model = Maze_model()
  change_step = 3000
  n = 10
  runs = 5
  max_step = 6000
  cum_reward = np.zeros((runs, 2, max_step))
  for r in tqdm(range(runs), ncols=64):
    seed = np.random.randint(1, 10000)
    for q in range(2):
      change_flag = True
      if q == 1:
        maze_model.init_model(k)
      else:
        maze_model.init_model()
      maze = Maze(obstacles=obstacles, start_state=start_state)
      step = 0
      last_step = step
      reward = 0
      value = np.zeros((maze.height + 2, maze.width + 2, 4))
      ep_greedy = Epsilon_greedy(seed=seed)
      while step < max_step:
        if q == 0:
          step += dyna_Q(n, maze, ep_greedy, value, maze_model)
        else:
          step += dyna_Q_plus(n, maze, ep_greedy, value, maze_model)
        cum_reward[r, q, last_step: min(step, max_step)] = reward
        reward += 1
        last_step = step
        if change_flag and step > change_step:
          maze = Maze(obstacles=new_obstacles, start_state=start_state)
          change_flag = False
  cum_reward = cum_reward.mean(axis=0)
  plt.plot(cum_reward[0], label='dyna-Q')
  plt.plot(cum_reward[1], label='dyna-Q+')
  plt.xlabel('time steps')
  plt.ylabel('cumulative reward')
  plt.legend()

  plt.savefig('../images/figure_8_4.png')
  plt.close()


def example_8_4():
  pass


if __name__ == '__main__':
  # fig_8_2()
  # fig_8_4()
  # fig_8_5()
  example_8_4()
  # maze = Maze()
  # print(maze._maze)
