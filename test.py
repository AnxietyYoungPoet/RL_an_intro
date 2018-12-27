import numpy as np

behavior_policy = np.ones((10, 10, 2)) / 2.
behavior_policy[:, :, :] = np.random.binomial(1, 0.5)
print(2 * np.random.binomial(1, 0.5) - 1)
