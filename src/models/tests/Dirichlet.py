import numpy as np
from math import gamma
from operator import mul
from functools import reduce
class Dirichlet:
    def __init__(self, alpha):
        self._alpha = np.array(alpha)
        self._coef = gamma(np.sum(self._alpha)) / reduce(mul, [gamma(a) for a in self._alpha])

    def pdf(self, x):
        """Returns pdf value for `x`. """
        return self._coef * reduce(mul, [xx ** (aa - 1) for (xx, aa) in zip(x, self._alpha)])

if __name__ == '__main__':
    #alpha = (0.2, 0.2, 0.25)
    x = [[7,10,2,45,10,2,9,3,9,0], [1,2,1,4,5,7,6,4,5,6] , [9,5,3,0,0,6,2,7,8,5]]
    
    alpha = []
    for x_ in x:
        alpha.append(np.mean(x_))
    dirichlet = Dirichlet(alpha)
    print(dirichlet._alpha)
    print(dirichlet._coef)
    print(dirichlet.pdf(x))