import random
import numpy as np
import pandas as pd
from sklearn.utils._random import sample_without_replacement


def _generate_indices(bootstrap, n_population, n_samples):
    """Draw randomly sampled indices."""
    # Draw sample indices
    randomlist = random.sample(range(10, 30), 5)
    if bootstrap:
        indices = random.randint(0, n_samples)
    else:
        indices = sample_without_replacement(
            20, n_samples, random_state=42
        )

    return indices
path = '../dataset/'
x = [1,2,4,3,5,6,7,8,9,10,1,2,3,4]
data = pd.read_csv(path+'adult.data.csv')
#PAY_AMT4 = data.PAY_AMT4.values.tolist()
capital_gain = data['capital-gain'].values.tolist()
print(np.unique(capital_gain))
#print(np.unique(capital_gain))
n_25 = np.percentile(capital_gain, 25)
n_50 = np.percentile(capital_gain,50)
n_75 = np.percentile(capital_gain,75)
print(n_25, n_50, n_75)

import itertools

#print(list(itertools.permutations(capital_gain)))