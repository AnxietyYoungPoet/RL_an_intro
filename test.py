import numpy as np

behavior_policy = np.ones((10, 10, 2)) / 2.
behavior_policy[:, :, :] = np.random.binomial(1, 0.5)
print(np.random.randint(0, 2, (10, 10, 2)))
