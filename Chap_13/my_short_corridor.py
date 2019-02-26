from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')


def true_value(p):
  return (2 * p - 4) / (p * (1 - p))


class ShortCorridor(object):
  def __init__(self):
    self.reset()
  
  def reset(self):
    self.state = 0
  
  def transition(self, action):
    if self.state == 1:
      self.state -= action
    else:
      self.state = max(0, self.state + action)
    if self.state == 3:
      return 0, True
    else:
      return -1, False


def softmax(x):
  t = np.exp(x - np.max(x))
  return t / np.sum(t)


class ReinforceAgent(object):
  def __init__(self, alpha, gamma):
    self.theta = np.array([-1.47, 1.47])
    self.alpha = alpha
    self.gamma = gamma
    self.x = np.array([[0, 1], [1, 0]])
    self.rewards = []
    self.actions = []
  
  def get_pi(self):
    scores  = np.dot(self.theta, self.x)
    pmf = softmax(scores)
    imin = np.argmin(pmf)
    epsilon = 0.05

    if pmf[imin] < epsilon:
      pmf[:] = 1 - epsilon
      pmf[imin] = epsilon

    return pmf
  
  def get_action(self):
    pmf = self.get_pi()
    random = np.random.uniform()
    if random < pmf[0]:
      action = -1
    else:
      action = 1
    return action
  
  def record(self, reward, action):
    self.rewards.append(reward)
    self.actions.append(action)
  
  def learn(self):
    G = np.zeros_like(self.rewards)
    G[-1] = self.rewards[-1]

    for i in range(2, len(G) + 1):
      G[-i] = self.gamma * G[-i + 1] + self.rewards[-i]
    
    for i in range(len(G)):
      j = int((self.actions[i] + 1) / 2)
      pmf = self.get_pi()
      grand_ln_pi = self.x[:, j] - np.dot(self.x, pmf)
      delta = self.alpha * G[i] * grand_ln_pi

      self.theta += delta
    
    self.rewards = []
    self.actions = []


class ReinforceBaselineAgent(ReinforceAgent):
  def __init__(self, alpha, gamma, alpha_w):
    super(ReinforceBaselineAgent, self).__init__(alpha, gamma)
    self.alpha_w = alpha_w
    self.w = 0
  
  def learn(self):
    G = np.zeros_like(self.rewards)
    G[-1] = self.rewards[-1]

    for i in range(2, len(G) + 1):
      G[-i] = self.gamma * G[-i + 1] + self.rewards[-i]
    
    for i in range(len(G)):
      self.w += self.alpha_w * (G[i] - self.w)

      j = int((self.actions[i] + 1) / 2)
      pmf = self.get_pi()
      grand_ln_pi = self.x[:, j] - np.dot(self.x, pmf)
      delta = self.alpha * (G[i] - self.w) * grand_ln_pi

      self.theta += delta

    self.rewards = []
    self.actions = []


def trial(num_episodes, agent_generator):
  env = ShortCorridor()
  agent = agent_generator()

  rewards = np.zeros(num_episodes)
  for episode_idx in range(num_episodes):
    rewards_sum = 0
    reward = None
    env.reset()

    while True:
      action = agent.get_action()
      reward, episode_end = env.transition(action)
      agent.record(reward, action)
      rewards_sum += reward

      if episode_end:
        agent.learn()
        break
      
    rewards[episode_idx] = rewards_sum
  return rewards


def example_13_1():
  epsilon = 0.05
  fig, ax = plt.subplots(1, 1)

  # Plot a graph
  p = np.linspace(0.01, 0.99, 100)
  y = true_value(p)
  ax.plot(p, y, color='red')

  # Find a maximum point, can also be done analytically by taking a derivative
  imax = np.argmax(y)
  pmax = p[imax]
  ymax = y[imax]
  ax.plot(pmax, ymax, color='green', marker="*",
          label="optimal point: f({0:.2f}) = {1:.2f}".format(pmax, ymax))

  # Plot points of two epsilon-greedy policies
  ax.plot(epsilon, true_value(epsilon), color='magenta',
          marker="o", label="epsilon-greedy left")
  ax.plot(1 - epsilon, true_value(1 - epsilon), color='blue',
          marker="o", label="epsilon-greedy right")

  ax.set_ylabel("Value of the first state")
  ax.set_xlabel("Probability of the action 'right'")
  ax.set_title("Short corridor with switched actions")
  ax.set_ylim(ymin=-105.0, ymax=5)
  ax.legend()

  plt.savefig('../images/example_13_1.png')
  plt.close()


def figure_13_1():
  num_trials = 30
  num_episodes = 1000
  alpha = 2e-4
  gamma = 1

  rewards = np.zeros((num_trials, num_episodes))
  def agent_generator(): return ReinforceAgent(alpha=alpha, gamma=gamma)

  for i in tqdm(range(num_trials)):
    reward = trial(num_episodes, agent_generator)
    rewards[i, :] = reward

  plt.plot(np.arange(num_episodes) + 1, -11.6 * np.ones(num_episodes),
           ls='dashed', color='red', label='-11.6')
  plt.plot(np.arange(num_episodes) + 1, rewards.mean(axis=0), color='blue')
  plt.ylabel('total reward on episode')
  plt.xlabel('episode')
  plt.legend(loc='lower right')

  plt.savefig('../images/figure_13_1.png')
  plt.close()


def figure_13_2():
  num_trials = 30
  num_episodes = 1000
  alpha = 2e-4
  gamma = 1
  agent_generators = [lambda: ReinforceAgent(alpha=alpha, gamma=gamma),
                      lambda: ReinforceBaselineAgent(alpha=alpha, gamma=gamma, alpha_w=alpha*100)]
  labels = ['Reinforce with baseline',
            'Reinforce without baseline']

  rewards = np.zeros((len(agent_generators), num_trials, num_episodes))

  for agent_index, agent_generator in enumerate(agent_generators):
    for i in tqdm(range(num_trials)):
      reward = trial(num_episodes, agent_generator)
      rewards[agent_index, i, :] = reward

  plt.plot(np.arange(num_episodes) + 1, -11.6 * np.ones(num_episodes),
           ls='dashed', color='red', label='-11.6')
  for i, label in enumerate(labels):
    plt.plot(np.arange(num_episodes) + 1, rewards[i].mean(axis=0), label=label)
  plt.ylabel('total reward on episode')
  plt.xlabel('episode')
  plt.legend(loc='lower right')

  plt.savefig('../images/figure_13_2.png')
  plt.close()

    
if __name__ == "__main__":
    # example_13_1()
    # figure_13_1()
    figure_13_2()
