from itertools import combinations
import numpy as np

def sigma_(p, q, alpha=0.01):
    ydiff = [abs(a-b) for a,b in zip(p,q)]
    _ydiff = [1/(abs(a - b)) for a, b in zip(p, q) if abs(a-b) > 0]

    perc_ydiff = np.percentile(ydiff, (1-alpha)*100)
    perc_ydiff_ = np.percentile(_ydiff, (1 - alpha) * 100)
    ydiff_contrained = [a for a in ydiff if a <= perc_ydiff]
    _ydiff_contrained = [a for a in _ydiff if a <= perc_ydiff_]
    wc_ = max(ydiff) * max(_ydiff)
    wc = max(ydiff_contrained) * max(_ydiff_contrained)
    print('Worse case distance: ', wc_, wc)
    print('ydiff: ', ydiff_contrained, _ydiff_contrained)
    print('ydiff: ', sorted(ydiff), sorted(_ydiff))
    #for i, j in combinations(range(len(p)), 2):
    #    print(i, j)
if __name__ == '__main__':
    y = [0.4, 0.45, 0.6, 0.3, 0.7, 0.8, 0.1, 0.0, 0.85]
    y_ = [0.2, 0.47, 0.65, 0.4, 0.82, 0.8, 0.15, 0.28, 0.9]

    #print(np.percentile(y, 80), max(y))
    #print(np.percentile(y_, 80), max(y_))
    sigma_(y, y_)