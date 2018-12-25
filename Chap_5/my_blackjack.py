import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

ACTION_HIT = 0
ACTION_STICK = 1
ACTIONS = [0, 1]  # hit ans strike


class Cards(object):
  def __init__(self):
    self.length = 5000000
    self._load_buffer()
    self.current = 0

  def _load_buffer(self):
    self.card_buffer = np.clip(np.random.choice(range(1, 14), self.length), 0, 10)

  def get_card(self, num=1):
    if self.current + num > self.length:
      self.current = 0
      self._load_buffer()
    value = self.card_buffer[self.current: self.current + num]
    if num == 1:
      value = value[0]
    self.current += num
    return value


def get_random_starts(episodes):
  player_sum = np.random.randint(12, 22, size=(episodes, 1))
  dealer_show = np.random.randint(1, 11, size=(episodes, 1))
  usable = np.random.randint(0, 2, size=(episodes, 2))
  starts = np.hstack((player_sum, dealer_show, usable))
  return starts


def play(policy, cards, start=None):
  dealer_init_cards = cards.get_card(2)
  dealer_show = dealer_init_cards[-1]
  usable_ace = False
  ace = False
  trajectory = []
  if start is None:
    player_init_cards = cards.get_card(2)
    player_sum = np.sum(player_init_cards)
    if 1 in player_init_cards:
      ace = True
    if ace and (player_sum + 10 <= 21):
      usable_ace = True
    while player_sum + usable_ace * 10 < 12:
      hit_card = cards.get_card()
      player_sum += hit_card
      ace = ace or hit_card == 1
      usable_ace = ace and (player_sum + 10 <= 21)
    action = policy[player_sum + usable_ace * 10 - 12, dealer_show - 1, int(usable_ace)]
  else:
    player_sum, dealer_show, usable_ace, action = start
    player_sum -= usable_ace * 10
    dealer_init_cards[-1] = dealer_show

  state = (usable_ace, player_sum, dealer_show)

  player_bust = False
  while True:
    trajectory.append((state, action))
    if action == ACTION_STICK:
      break
    hit_card = cards.get_card()
    player_sum += hit_card
    if player_sum > 21:
      player_bust = True
      break
    usable_ace = usable_ace and (player_sum + 10 <= 21)
    state = (usable_ace, player_sum, dealer_show)
    action = policy[player_sum + usable_ace * 10 - 12, dealer_show - 1, int(usable_ace)]
  if player_bust:
    return -1, trajectory

  dealer_sum = np.sum(dealer_init_cards)
  dealer_ace = False
  dealer_usable_ace = False
  if 1 in dealer_init_cards:
    dealer_ace = True
  if dealer_ace and dealer_sum + 10 <= 21:
    dealer_usable_ace = True
  while dealer_sum + dealer_usable_ace * 10 < 17:
    hit_card = cards.get_card()
    dealer_sum += hit_card
    dealer_ace = dealer_ace or (hit_card == 1)
    dealer_usable_ace = dealer_ace and dealer_sum + 10 <= 21
  if (dealer_sum > 21) or (
    dealer_sum + dealer_usable_ace * 10 < player_sum + usable_ace * 10):
    reward = 1
  elif dealer_sum + dealer_usable_ace * 10 == player_sum + usable_ace * 10:
    reward = 0.5
  else:
    reward = -1
  return reward, trajectory


def monte_carlo_on_policy(episodes):
  cards = Cards()
  policy = np.zeros((10, 10, 2))
  policy[8:, :, :] = 1
  usable_ace_value = np.zeros((10, 10))
  usable_ace_count = np.zeros_like(usable_ace_value)
  no_usable_ace_value = np.zeros((10, 10))
  no_usable_ace_count = np.zeros_like(no_usable_ace_value)
  for i in tqdm(range(episodes), ncols=64):
    reward, trajectory = play(policy, cards)
    for state, _ in trajectory:
      usable_ace, player_sum, dealer_show = state
      player_sum = player_sum + usable_ace * 10 - 12
      dealer_show -= 1
      if usable_ace:
        usable_ace_value[player_sum, dealer_show] += reward
        usable_ace_count[player_sum, dealer_show] += 1
      else:
        no_usable_ace_value[player_sum, dealer_show] += reward
        no_usable_ace_count[player_sum, dealer_show] += 1
  return usable_ace_value / (usable_ace_count + 1e-5), \
      no_usable_ace_value / (no_usable_ace_count + 1e-5)


def monte_carlo_ES(episodes):
  cards = Cards()
  policy = np.zeros((10, 10, 2))
  # policy[8:, :, :] = 1
  usable_ace_value = np.zeros((10, 10, 2))
  usable_ace_count = np.zeros_like(usable_ace_value)
  no_usable_ace_value = np.zeros((10, 10, 2))
  no_usable_ace_count = np.zeros_like(no_usable_ace_value)

  random_starts = get_random_starts(episodes)
  for random_start in tqdm(random_starts, ncols=64):
    reward, trajectory = play(policy, cards, random_start)
    for state, action in trajectory:
      usable_ace, player_sum, dealer_show = state
      ps = player_sum + usable_ace * 10 - 12
      ds = dealer_show - 1
      if usable_ace:
        usable_ace_value[ps, ds, int(action)] += reward
        usable_ace_count[ps, ds, int(action)] += 1
        policy[ps, ds, 1] = np.argmax(
          usable_ace_value[ps, ds] / (usable_ace_count[ps, ds] + 1e-5))
      else:
        no_usable_ace_value[ps, ds, int(action)] += reward
        no_usable_ace_count[ps, ds, int(action)] += 1
        policy[ps, ds, 0] = np.argmax(
          no_usable_ace_value[ps, ds] / (no_usable_ace_count[ps, ds] + 1e-5))
  
  return usable_ace_value / (usable_ace_count + 1e-5), \
      no_usable_ace_value / (no_usable_ace_count + 1e-5)


def fig_5_1():
  states_usable_ace_1, states_no_usable_ace_1 = monte_carlo_on_policy(10000)
  states_usable_ace_2, states_no_usable_ace_2 = monte_carlo_on_policy(500000)

  states = [states_usable_ace_1,
            states_usable_ace_2,
            states_no_usable_ace_1,
            states_no_usable_ace_2]

  titles = ['Usable Ace, 10000 Episodes',
            'Usable Ace, 500000 Episodes',
            'No Usable Ace, 10000 Episodes',
            'No Usable Ace, 500000 Episodes']

  _, axes = plt.subplots(2, 2, figsize=(40, 30))
  plt.subplots_adjust(wspace=0.1, hspace=0.2)
  axes = axes.flatten()

  for state, title, axis in zip(states, titles, axes):
    fig = sns.heatmap(np.flipud(state), cmap="YlGnBu", ax=axis, xticklabels=range(1, 11),
                      yticklabels=list(reversed(range(12, 22))))
    fig.set_ylabel('player sum', fontsize=30)
    fig.set_xlabel('dealer showing', fontsize=30)
    fig.set_title(title, fontsize=30)

  plt.savefig('../images/figure_5_1.png')
  plt.close()


def fig_5_2():
  states_usable_ace, states_no_usable_ace = monte_carlo_ES(10000000)

  state_value_no_usable_ace = np.max(states_no_usable_ace[:, :, :], axis=-1)
  state_value_usable_ace = np.max(states_usable_ace[:, :, :], axis=-1)

  # get the optimal policy
  action_no_usable_ace = np.argmax(states_no_usable_ace[:, :, :], axis=-1)
  action_usable_ace = np.argmax(states_usable_ace[:, :, :], axis=-1)

  images = [action_usable_ace,
            state_value_usable_ace,
            action_no_usable_ace,
            state_value_no_usable_ace]

  titles = ['Optimal policy with usable Ace',
            'Optimal value with usable Ace',
            'Optimal policy without usable Ace',
            'Optimal value without usable Ace']

  _, axes = plt.subplots(2, 2, figsize=(40, 30))
  plt.subplots_adjust(wspace=0.1, hspace=0.2)
  axes = axes.flatten()

  for image, title, axis in zip(images, titles, axes):
    fig = sns.heatmap(np.flipud(image), cmap="YlGnBu", ax=axis, xticklabels=range(1, 11),
                      yticklabels=list(reversed(range(12, 22))))
    fig.set_ylabel('player sum', fontsize=30)
    fig.set_xlabel('dealer showing', fontsize=30)
    fig.set_title(title, fontsize=30)

  plt.savefig('../images/figure_5_2.png')
  plt.close()


if __name__ == '__main__':
  # fig_5_1()
  fig_5_2()
