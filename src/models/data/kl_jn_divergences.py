import math
from math import log2

import numpy as np
import scipy
from numpy.core import linspace
from scipy.special import rel_entr
from scipy.stats import norm, gaussian_kde, entropy
from matplotlib import pyplot as plt
import tensorflow as tf
import seaborn as sns

MAX_ENTROPY_ALLOWED = 1e6  # A hack to never deal with inf entropy values that happen when the PDFs don't intersect

sns.set()

class JSDivergence:
    # Calculates the JSD for multiple probability distributions
    def jsd(self, prob_dists):
        weight = 1 / len(prob_dists)  # Set weights to be uniform
        js_left = np.zeros(len(prob_dists[0]))
        js_right = 0
        for pd in prob_dists:
            js_left += pd * weight
            js_right += weight * self.entropy(pd, normalize=False)

        jsd = self.entropy(js_left, normalize=False) - js_right
        return jsd

    # Entropy function
    def entropy(self, prob_dist, normalize=True):
        entropy = -sum([p * math.log2(p) for p in prob_dist if p != 0])
        if normalize:
            max_entropy = math.log2(prob_dist.shape[0])
            return entropy / max_entropy
        return entropy
def compute_js_divergence(p,q):
    n1 = len(p)
    n2 = len(q)
    if n1 <= 1 or n2 <= 1:
        return 0.0, 0.0
    # Calculate the interval to be analyzed further
    a = min(min(p), min(q))
    b = max(max(p), max(q))

    # Plot the PDFs
    max_bins = max(n1, n2)
    #lin = linspace(a, b, max(n1, n2))
    # p = pdf1.pdf(lin)
    # q = pdf2.pdf(lin)
    # p = norm.pdf(p, loc=0, scale=1)
    # q = norm.pdf(q, loc=0, scale=1)

    # Construct empirical PDF with these two samples
    hist1 = np.histogram(p, bins=max_bins) # bins=10
    hist1_dist = scipy.stats.rv_histogram(hist1)
    hist2 = np.histogram(q, bins=max_bins) # bins=10
    hist2_dist = scipy.stats.rv_histogram(hist2)
    X = np.linspace(a, b, max_bins)
    Y1 = hist1_dist.pdf(X)
    Y2 = hist2_dist.pdf(X)

    meanP = np.mean(p)
    stdP = np.std(p)

    meanQ = np.mean(q)
    stdQ = np.std(q)

    pdf_normalP = normal_pdf(p, meanP, covP)
    pdf_normalQ = normal_pdf(q, meanQ, covQ)

    # compute KL divergence between Y1 and Y2
    kl_diverg = scipy.stats.entropy(Y1, Y2, base=2)

    # Obtain point-wise mean of the two PDFs Y1 and Y2, denote it as M
    M = (Y1 + Y2) / 2
    # Compute Kullback-Leibler divergence between Y1 and M
    d1 = scipy.stats.entropy(Y1, M, base=2)
    # d1 = 0.406
    # Compute Kullback-Leibler divergence between Y2 and M
    d2 = scipy.stats.entropy(Y2, M, base=2)
    # d2 = 0.300

    # Take the average of d1 and d2
    # we get the symmetric Jensen-Shanon divergence
    js_dv = (d1 + d2) / 2
    # js_dv = 0.353
    # Jensen-Shanon distance is the square root of the JS divergence
    js_distance = np.sqrt(js_dv)
    # js_distance = 0.594
    # Check it against scipy's calculation
    js_distance_scipy = scipy.spatial.distance.jensenshannon(Y1, Y2)
    # js_distance_scipy = 0.493
    return js_distance_scipy, js_distance #, js_distance_scipy



def kl_divergence_sklearn(p_samples, q_samples):
    n1 = len(p_samples)
    n2 = len(q_samples)
    if n1 == 0 or n2 == 0:
        return 0.0, None,None, None
    #try:
    # Estimate the PDFs using Gaussian KDE
    pdf1 = gaussian_kde(p_samples)
    pdf2 = gaussian_kde(q_samples)
    # Calculate the interval to be analyzed further
    a = min(min(p_samples), min(q_samples))
    b = max(max(p_samples), max(q_samples))

    # Plot the PDFs
    lin = linspace(a, b, max(n1, n2))
    p = pdf1.pdf(lin)
    q = pdf2.pdf(lin)

    return min(MAX_ENTROPY_ALLOWED, entropy(p, q))


def kl_divergence_1(p, q):
 return np.sum(np.where(p != 0, p * np.log(p / q), 0))

#todo: https://yongchaohuang.github.io/2020-07-08-kl-divergence/

'''The log can be base-2 to give units in “bits,” or the natural 
logarithm base-e with units in “nats.” When the score is 0, 
it suggests that both distributions are identical, otherwise the score is positive.
This sum (or integral in the case of continuous random variables) 
will always be positive, by the Gibbs inequality.'''

'''We can develop a function to calculate the KL divergence 
between the two distributions. We will use log base-2 
to ensure the result has units in bits'''
# calculate the kl divergence
def kl_divergence(p, q):
    n1 = len(p)
    n2 = len(q)
    if n1 <= 1 or n2 <= 1:
        return 0.0
    # Calculate the interval to be analyzed further
    a = min(min(p), min(q))
    b = max(max(p), max(q))

    # Plot the PDFs
    max_bins = max(n1, n2)
    # lin = linspace(a, b, max(n1, n2))
    # p = pdf1.pdf(lin)
    # q = pdf2.pdf(lin)
    # p = norm.pdf(p, loc=0, scale=1)
    # q = norm.pdf(q, loc=0, scale=1)

    # Construct empirical PDF with these two samples
    hist1 = np.histogram(p, bins=max_bins)  # bins=10
    hist1_dist = scipy.stats.rv_histogram(hist1)
    hist2 = np.histogram(q, bins=max_bins)  # bins=10
    hist2_dist = scipy.stats.rv_histogram(hist2)
    X = np.linspace(a, b, max_bins)
    Y1 = hist1_dist.pdf(X)
    Y2 = hist2_dist.pdf(X)
    # Compute Kullback-Leibler divergence between Y1 and M
    d1 = scipy.stats.entropy(Y1, Y2, base=2)
    return d1

def kl_divergence_(p, q):
    #pdf1 = gaussian_kde(p_samples)
    #pdf2 = gaussian_kde(q_samples)

    # Calculate the interval to be analyzed further
    #a = min(min(p), min(q))
    #b = max(max(p), max(q))


    # Plot the PDFs
    #lin = linspace(a, b, max(n1, n2))
    #p = pdf1.pdf(lin)
    #q = pdf2.pdf(lin)
    #p = norm.pdf(p, loc=0, scale=1)
    #q = norm.pdf(q, loc=0, scale=1)

    #p = abs(p)
    #q = abs(q)
    #p = norm.pdf(p, 0, 2)
    #q = norm.pdf(q, 2, 2)

    #p = norm.pdf(p)
    #q = norm.pdf(q)
    #print('p & q: ', p, q)
    #for i in range(len(p)):
    #    print(i, p[i], q[i], np.log2(p[i] / q[i]))
    #return sum(p[i] * log2(p[i] / q[i]) for i in range(len(p)) if (p[i] / q[i]) >=0)
    n1 = len(p)
    n2 = len(q)
    if n1 <= 1 or n2 <= 1:
        return 0.0
    # Estimate the PDFs using Gaussian KDE
    print(p, q)
    pdf1 = gaussian_kde(p)
    pdf2 = gaussian_kde(q)
    # Calculate the interval to be analyzed further
    a = min(min(p), min(q))
    b = max(max(p), max(q))

    # Plot the PDFs
    lin = linspace(a, b, max(n1, n2))
    #p = pdf1.pdf(lin)
    #q = pdf2.pdf(lin)
    p = np.array(p)
    q = np.array(q)
    #q = np.array([q[i] for i in range(p) if p[i] != 0])
    #return np.sum(np.where(p != 0, p * np.log(p / q), 0))
    #return np.sum(rel_entr(p, q))
    #return  scipy.stats.entropy(Y1, M, base=2) ## Compute Kullback-Leibler divergence between Y1 and M
    #return min(MAX_ENTROPY_ALLOWED, entropy(p, q))

# Not that KL divergence is not symetric implying that
# calculate KL (P || Q) is not the same with # calculate (Q || P)

'''The SciPy library provides the kl_div() function for calculating 
the KL divergence, although with a different definition as defined here. 
It also provides the rel_entr() function for calculating the relative entropy, 
which matches the definition of KL divergence here.
The rel_entr() function calculation uses the natural logarithm instead of 
log base-2 so the units are in nats instead of bits.'''


## TODO: Jensen-Shannon Divergence The concept

'''The Jensen-Shannon divergence, or JS divergence for short, 
is another way to quantify the difference (or similarity) between two probability distributions
It uses the KL divergence to calculate a normalized score that is symmetrical. 
This means that the divergence of P from Q is the same as Q from P:
JS(P || Q) == JS(Q || P)

The JS divergence can be calculated as follows:
JS(P || Q) = 1/2 * KL(P || M) + 1/2 * KL(Q || M)
Where M is calculated as:
M = 1/2 * (P + Q)

It is more useful as a measure as it provides a smoothed and normalized 
version of KL divergence, with scores between 0 (identical) and 1 (maximally different), 
when using the base-2 logarithm.
The square root of the score gives a quantity referred to as the Jensen-Shannon distance, 
or JS distance for short.'''

## define a function to calculate the JS divergence that uses the kl_divergence() function prepared in the previous section:

def js_divergence(p, q):
    #print('p: ', p)
    #print('q: ', q)
    m = (p+q)
    #print(m)
    m = [i*0.5 for i in m]
    #print('m: ', m)
    #m = (p + q)/2 # 0.5*(p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

## calculate JS(P || Q) and calculate JS(Q || P)
## the expected results should be the same,  meaning JS divergence is symmetrical

def kl_js_divergence_metrics(p, q):
    js_divergence = compute_js_divergence(p,q)
    print('compute_js_divergence: ', js_divergence)
    return js_divergence

    #return kl_divergence(p, q), js_divergence(p, q)