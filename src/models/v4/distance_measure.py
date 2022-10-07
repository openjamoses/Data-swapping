import numpy as np
import math
from scipy import integrate, stats, spatial
import torch
#from layers import SinkhornDistance
import pandas as pd

import math
SQRT_TWO_PI = math.sqrt(2 * math.pi)
def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return stats.norm.pdf(x)# (math.exp(-(x-mu) ** 2 / 2 / sigma ** 2) / (SQRT_TWO_PI * sigma))

def f(z, μ, Σ):
    print('z', z)
    #detZ = np.linalg.det(z)
    #print('detZ', detZ)
    #μ = np.mean(z)
    #print(z.shape, μ.shape, Σ.shape)
    """
    The density function of multivariate normal distribution.

    Parameters
    ---------------
    z: ndarray(float, dim=2)
        random vector, N by 1
    μ: ndarray(float, dim=1 or 2)
        the mean of z, N by 1
    Σ: ndarray(float, dim=2)
        the covarianece matrix of z, N by 1
    """

    z = np.atleast_2d(z)
    μ = np.atleast_2d(μ)
    Σ = np.atleast_2d(Σ)

    N = z.size

    #print(z)
    det_ = np.linalg.det(Σ)
    temp1 = det_ ** (-1/2)
    #print('inverse matrix: ', np.linalg.det(Σ))

    if det_ != 0:
        temp2 = np.exp(-.5 * (z - μ).T @ np.linalg.inv(Σ) @ (z - μ))
    else:
        temp2 = np.exp(-.5 * (z - μ).T @ np.linalg.pinv(Σ) @ (z - μ))
    #temp2 = np.exp(-.5 * (z - μ).T @ np.linalg.inv(Σ) @ (z - μ))

    return (2 * np.pi) ** (-N/2) * temp1 * temp2

class DistanceMeasure:
    @staticmethod
    def hellinger_discrete_1d(p, q):
        p = [x / np.sum(p) for x in p]
        q = [x / np.sum(q) for x in q]
        #p = stats.norm.cdf(p, np.mean(p), np.std(p))  # .pdf(p)
        #q = stats.norm.cdf(q, np.mean(q), np.std(q))  # stats.uniform.pdf(q)
        """Hellinger distance between two discrete distributions.
           In pure Python.
           Some improvements.
        """
        return math.sqrt(sum([(math.sqrt(p_i) - math.sqrt(q_i)) ** 2 for p_i, q_i in zip(p, q)]) / 2)
    @staticmethod
    def hellinger_continous_1d(p,q):
        p = normal_pdf(p) #[x/np.sum(p) for x in p]
        q = normal_pdf(q) #[x / np.sum(q) for x in q]
        #p = stats.norm.cdf(p, np.mean(p), np.std(p))#.pdf(p)
        #q = stats.norm.cdf(q, np.mean(q), np.std(q)) #stats.uniform.pdf(q)
        return 1/np.sqrt(2)*np.linalg.norm(np.sqrt(p)-np.sqrt(q))

    @staticmethod
    def hellinger_multivariate(X, Y):
        """ Calculates Hellinger distance between 2 multivariate normal distribution
             X = X(x1, x2)
             Y = Y(y1, y2)
             The definition can be found at https://en.wikipedia.org/wiki/Hellinger_distance
        """
        if len(X) < 2 or len(Y) < 2:      return 1.

        meanX = np.mean(X, axis=0)
        covX = np.cov(X, rowvar=0)
        detX = np.linalg.det(covX)

        meanY = np.mean(Y, axis=0)
        covY = np.cov(Y, rowvar=0)
        detY = np.linalg.det(covY)

        detXY = np.linalg.det((covX + covY) / 2)
        if (np.linalg.det(covX + covY) / 2) != 0:
            covXY_inverted = np.linalg.inv((covX + covY) / 2)
        else:
            covXY_inverted = np.linalg.pinv((covX + covY) / 2)
        dist = 1. - (detX ** .25 * detY ** .25 / detXY ** .5) * np.exp(
            -.125 * np.dot(np.dot(np.transpose(meanX - meanY), covXY_inverted), (meanX - meanY)))

        #dist_2 = 1/np.sqrt(2)*np.linalg.norm(np.sqrt(detX)-np.sqrt(detY))

        
        #dis_3 = DistanceMeasure.hellinger_discrete_1d(X,Y)
        #dis_4 = DistanceMeasure.hellinger_continous_1d(X, Y)
        #print('hellinger_multivariate: ', dist, dist_2,dis_3, dis_4)
        #print('hellinger_multivariate: ', meanX, meanY, covXY_inverted)'''
        #X, Y = np.array(X).flatten(), np.array(Y).flatten()
        #dis_3 = DistanceMeasure.hellinger_discrete_1d(X, Y)
        #return dist






        return min(max(dist, 0.), 1.)
    @staticmethod
    def js_divergence(p, q):
        n1 = len(p)
        n2 = len(q)
        if n1 <= 1 or n2 <= 1:
            return 0.0, 0.0
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
        hist1_dist = stats.rv_histogram(hist1)
        hist2 = np.histogram(q, bins=max_bins)  # bins=10
        hist2_dist = stats.rv_histogram(hist2)
        X = np.linspace(a, b, max_bins)
        Y1 = hist1_dist.pdf(X)
        Y2 = hist2_dist.pdf(X)

        #pdf_normalP = normal_pdf(p, np.mean(p), np.std(p))
        #pdf_normalQ = normal_pdf(q, np.mean(q), np.std(q))
        #kl_diverg1 = stats.entropy(pdf_normalP, pdf_normalQ, base=2)

        #print('pdf_normalQ: ', pdf_normalP, pdf_normalQ)

        # compute KL divergence between Y1 and Y2
        kl_diverg = stats.entropy(Y1, Y2, base=2)

        #print('KL compare: ', kl_diverg1, kl_diverg)

        # Obtain point-wise mean of the two PDFs Y1 and Y2, denote it as M
        M = (Y1 + Y2) / 2
        # Compute Kullback-Leibler divergence between Y1 and M
        d1 = stats.entropy(Y1, M, base=2)
        # d1 = 0.406
        # Compute Kullback-Leibler divergence between Y2 and M
        d2 = stats.entropy(Y2, M, base=2)
        # d2 = 0.300

        # Take the average of d1 and d2
        # we get the symmetric Jensen-Shanon divergence
        js_dv = (d1 + d2) / 2
        # js_dv = 0.353
        # Jensen-Shanon distance is the square root of the JS divergence
        js_distance = np.sqrt(js_dv)
        # js_distance = 0.594
        # Check it against scipy's calculation
        #js_distance_scipy = spatial.distance.jensenshannon(Y1, Y2)
        # js_distance_scipy = 0.493
        return kl_diverg, js_distance  # , js_distance_scipy

    @staticmethod
    def js_divergence_2d(p, q):
        p = np.array(p)
        q = np.array(q)
        n1 = len(p)
        n2 = len(q)
        if n1 <= 1 or n2 <= 1:
            return 0.0, 0.0
        # Calculate the interval to be analyzed further
        a = min(min(p.flatten()), min(q.flatten()))
        b = max(max(p.flatten()), max(q.flatten()))
        # Plot the PDFs
        max_bins = max(n1, n2)
        # lin = linspace(a, b, max(n1, n2))
        # p = pdf1.pdf(lin)
        # q = pdf2.pdf(lin)
        # p = norm.pdf(p, loc=0, scale=1)
        # q = norm.pdf(q, loc=0, scale=1)

        # Construct empirical PDF with these two samples
        hist1 = np.histogram(p, bins=max_bins)  # bins=10
        hist1_dist = stats.rv_histogram(hist1)
        hist2 = np.histogram(q, bins=max_bins)  # bins=10
        hist2_dist = stats.rv_histogram(hist2)
        X = np.linspace(a, b, max_bins)
        Y1 = hist1_dist.pdf(X)
        Y2 = hist2_dist.pdf(X)

        meanp = np.mean(p)#, axis=0
        covp = np.cov(p)#, rowvar=0
        #detX = np.linalg.det(covX)

        meanq = np.mean(q) #, axis=0
        covq = np.cov(q) #, rowvar=0
        #detY = np.linalg.det(covY)


        p_dist = f(p, meanp, covp)
        q_dist = f(q, meanq, covq)

        print('p_dist: ', p_dist)
        print('q_dist: ', q_dist)

        kl_diverg = stats.entropy(p_dist, q_dist, base=2)

        print('kl diverg manual: ', np.mean(kl_diverg))


        # compute KL divergence between Y1 and Y2
        kl_diverg = stats.entropy(Y1, Y2, base=2)

        print('kl diverg Automated: ', kl_diverg)

        # Obtain point-wise mean of the two PDFs Y1 and Y2, denote it as M
        M = (Y1 + Y2) / 2
        # Compute Kullback-Leibler divergence between Y1 and M
        d1 = stats.entropy(Y1, M, base=2)
        # d1 = 0.406
        # Compute Kullback-Leibler divergence between Y2 and M
        d2 = stats.entropy(Y2, M, base=2)
        # d2 = 0.300

        # Take the average of d1 and d2
        # we get the symmetric Jensen-Shanon divergence
        js_dv = (d1 + d2) / 2
        # js_dv = 0.353
        # Jensen-Shanon distance is the square root of the JS divergence
        js_distance = np.sqrt(js_dv)
        # js_distance = 0.594
        # Check it against scipy's calculation
        # js_distance_scipy = spatial.distance.jensenshannon(Y1, Y2)
        # js_distance_scipy = 0.493
        return kl_diverg, js_distance  # , js_distance_scipy
    @staticmethod
    def wasserstein_distance(p,q):
        print('p before: ', p)
        p = np.array(p)
        q = np.array(q)
        p = p.flatten()
        q = q.flatten()
        print('p before: ', p)
        p = stats.norm.cdf(p, np.mean(p), np.std(p))  # .pdf(p)
        q = stats.norm.cdf(q, np.mean(q), np.std(q))  # stats.uniform.pdf(q)
        #dis_ = stats.wasserstein_distance(p, q)
        try:
            dis_ = stats.wasserstein_distance(p,q)
        except Exception as e:
            dis_ = -1
        return dis_

    @staticmethod
    def wasserstein_distance_pdf(p, q):

        n1 = len(p)
        n2 = len(q)
        if n1 <= 1 or n2 <= 1:
            return 0.0, 0.0
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
        hist1_dist = stats.rv_histogram(hist1)
        hist2 = np.histogram(q, bins=max_bins)  # bins=10
        hist2_dist = stats.rv_histogram(hist2)
        X = np.linspace(a, b, max_bins)
        Y1 = hist1_dist.pdf(X)
        Y2 = hist2_dist.pdf(X)

        print(' Total pdf: ', np.sum(Y1), np.sum(Y2))

        #print('p before: ', p)
        #p = np.array(p)
        #q = np.array(q)
        #p = p.flatten()
        #q = q.flatten()
        #print('p before: ', p)
        #p = stats.norm.cdf(p, np.mean(p), np.std(p))  # .pdf(p)
        #q = stats.norm.cdf(q, np.mean(q), np.std(q))  # stats.uniform.pdf(q)
        # dis_ = stats.wasserstein_distance(p, q)
        try:
            p = [x / np.sum(p) for x in p]
            q = [x / np.sum(q) for x in q]

            dis_ = sum([abs(p_i - q_i) for p_i, q_i in zip(p, q)])
            dis_ = dis_*0.5
            #dis_ = stats.wasserstein_distance(Y1, Y2)

        except Exception as e:
            print('Error in wasserstein_distance_pdf: ', e)
            dis_ = -1
        return dis_
    @staticmethod
    def total_variation_distance(p,q):
        p = [x / np.sum(p) for x in p]
        q = [x / np.sum(q) for x in q]
        dis_ = sum([abs(p_i - q_i) for p_i, q_i in zip(p, q)])
        dis_ = dis_ * 0.5
        return dis_

    @staticmethod
    def wasserstein_distance_pdf2d(p, q):
        p = np.array(p)
        q = np.array(q)
        n1 = len(p)
        n2 = len(q)
        if n1 <= 1 or n2 <= 1:
            return 0.0, 0.0
        # Calculate the interval to be analyzed further
        a = min(min(p.flatten()), min(q.flatten()))
        b = max(max(p.flatten()), max(q.flatten()))
        # Plot the PDFs
        max_bins = max(n1, n2)
        # lin = linspace(a, b, max(n1, n2))
        # p = pdf1.pdf(lin)
        # q = pdf2.pdf(lin)
        # p = norm.pdf(p, loc=0, scale=1)
        # q = norm.pdf(q, loc=0, scale=1)

        # Construct empirical PDF with these two samples
        hist1 = np.histogram(p, bins=max_bins)  # bins=10
        hist1_dist = stats.rv_histogram(hist1)
        hist2 = np.histogram(q, bins=max_bins)  # bins=10
        hist2_dist = stats.rv_histogram(hist2)
        X = np.linspace(a, b, max_bins)
        Y1 = hist1_dist.pdf(X)
        Y2 = hist2_dist.pdf(X)

        # print('p before: ', p)
        # p = np.array(p)
        # q = np.array(q)
        # p = p.flatten()
        # q = q.flatten()
        # print('p before: ', p)
        # p = stats.norm.cdf(p, np.mean(p), np.std(p))  # .pdf(p)
        # q = stats.norm.cdf(q, np.mean(q), np.std(q))  # stats.uniform.pdf(q)
        # dis_ = stats.wasserstein_distance(p, q)
        try:
            #p = p.flatten()
            #q = q.flatten()
            #p = [x / np.sum(p) for x in p]
            #q = [x / np.sum(q) for x in q]

            dis_ = stats.wasserstein_distance(Y1, Y2)
        except Exception as e:
            print('Error in wasserstein_distance_pdf: ', e)
            dis_ = -1
        return dis_

    @staticmethod
    def cramers_v(p, q):
        if len(p) > 0:
            #print('error seems to be here 0..!')
            vals, count = stats.contingency.crosstab(p, q)

            #data_crosstab = pd.crosstab(p, q, margins=False)
            #print('error seems to be here 0 end..!')
            #print('count:', count, vals)
            #print('data_crosstab: ', data_crosstab)


            #print('error seems to be here 1..!')
            #assoc_1 = stats.contingency.association(count, method="cramer")
            assoc_2 = stats.contingency.association(count,  method="pearson")
            #print('error seems to be here 1 ends..!')



            # create 2x2 table
            #data = count #np.array([p, q])

            # Chi-squared test statistic, sample size, and minimum of rows and columns
            #print('error seems to be here 2..!')
            #X2 = stats.chi2_contingency(data, correction=False)[0]
            #print('error seems to be here 2 end..!')
            #n = np.sum(data)
            #minDim = min(data.shape) - 1

            # calculate Cramer's V
            #print('error seems to be here 3..!')
            #V = np.sqrt((X2 / n) / minDim)
            #print('error seems to be here 3 ends..!')

            #print('chi2_contingency: ', assoc_1, assoc_2, V)

            # display Cramer's V
            #print(V)
            #print(' --- coef: ', assoc_1, assoc_2)
            return assoc_2
        else:
            return -1