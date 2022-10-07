#from scipy.stats import uniform

from src.models.data.load_train_test import LoadTrainTest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import scipy
import math
from scipy import integrate, stats
def modify_df(x):
    modifications = 0.0213
    return x + modifications
class DistanceMeasure:
    @staticmethod
    def hellinger_distance(p, q):
        n1 = len(p)
        n2 = len(q)
        if n1 <= 1 or n2 <= 1:
            return 0.0
        # Calculate the interval to be analyzed further
        a = min(min(p), min(q))
        b = max(max(p), max(q))

        # Plot the PDFs
        max_bins = max(n1, n2)
        hist1 = np.histogram(p, bins=max_bins)  # bins=10
        hist1_dist = stats.rv_histogram(hist1)
        hist2 = np.histogram(q, bins=max_bins)  # bins=10
        hist2_dist = stats.rv_histogram(hist2)
        X = np.linspace(a, b, max_bins)
        Y1 = hist1_dist.pdf(X)
        Y2 = hist2_dist.pdf(X)
        # Compute Kullback-Leibler divergence between Y1 and M
        # distance = (1/math.sqrt(2) )* integrate((math.sqrt(Y1)-math.sqrt(Y2))**2)
        # d1 = scipy.stats.entropy(Y1, Y2, base=2)
        print([(math.sqrt(p_i) - math.sqrt(q_i)) ** 2 for p_i, q_i in zip(Y1, Y2)])
        return math.sqrt(integrate([(math.sqrt(p_i) - math.sqrt(q_i)) ** 2 for p_i, q_i in zip(Y1, Y2)]) / 2)

        # return distance

    @staticmethod
    def hellinger2(p, q):
        p = stats.norm.cdf(p, np.mean(p), np.std(p))  # .pdf(p)
        q = stats.norm.cdf(q, np.mean(q), np.std(q))  # stats.uniform.pdf(q)
        """Hellinger distance between two discrete distributions.
           In pure Python.
           Some improvements.
        """
        return math.sqrt(sum([(math.sqrt(p_i) - math.sqrt(q_i)) ** 2 for p_i, q_i in zip(p, q)]) / 2)
    @staticmethod
    def hellinger_continous(p,q):
        p = stats.norm.cdf(p, np.mean(p), np.std(p))#.pdf(p)
        q = stats.norm.cdf(q, np.mean(q), np.std(q)) #stats.uniform.pdf(q)
        return 1/np.sqrt(2)*np.linalg.norm(np.sqrt(p)-np.sqrt(q))

    @staticmethod
    def hellinger_dist(X, Y):
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
        #print(dist)
        return min(max(dist, 0.), 1.)
if __name__ == '__main__':
    data_name = 'adult'
    path = '../dataset/experiments/data/'
    load_data = LoadTrainTest(path)
    train, test, target_index = load_data.load_adult(data_name)

    age = test.age.values
    capital_gain = test['capital-gain'].values
    capital_loss = test['capital-loss'].values
    hours_per_week = test['hours-per-week'].values

    age_transformed = age*0.0213
    capital_gain_transformed = capital_gain * 0.0213
    capital_loss_transformed = capital_loss * 0.0213
    hours_per_week_transformed = hours_per_week * 0.0213

    D_test_original = test.to_numpy()
    D_test_transformed = test.copy()
    D_test_transformed['age'] = D_test_transformed.age.apply(modify_df).to_numpy()
    D_test_transformed2 = test.copy()
    D_test_transformed2['capital-gain'] = D_test_transformed2['capital-gain'].apply(modify_df).to_numpy()
    D_test_transformed3 = test.copy()
    D_test_transformed3['capital-loss'] = D_test_transformed3['capital-loss'].apply(modify_df).to_numpy()
    D_test_transformed4 = test.copy()
    D_test_transformed4['hours-per-week'] = D_test_transformed4['hours-per-week'].apply(modify_df).to_numpy()

    print('D_test_original: ', D_test_original)
    print('D_test_transformed: ', D_test_transformed)

    #distance =DistanceMeasure.hellinger_dist(D_test_original, D_test_transformed)
    print('Distance: ', DistanceMeasure.hellinger_dist(D_test_original, D_test_transformed))
    print('Distance2: ', DistanceMeasure.hellinger_dist(D_test_original, D_test_transformed2))
    print('Distance3: ', DistanceMeasure.hellinger_dist(D_test_original, D_test_transformed3))
    print('Distance4: ', DistanceMeasure.hellinger_dist(D_test_original, D_test_transformed4))

    print('Distance measure for univariate')
    print('Age: ', DistanceMeasure.hellinger_continous(age, age_transformed))
    print('capital_gain: ', DistanceMeasure.hellinger_continous(capital_gain, capital_gain_transformed))
    print('capital_loss: ', DistanceMeasure.hellinger_continous(capital_loss, capital_loss_transformed))
    print('hours_per_week: ', DistanceMeasure.hellinger_continous(hours_per_week, hours_per_week_transformed))


    print('Age: ', DistanceMeasure.hellinger2(age, age_transformed))
    print('capital_gain: ', DistanceMeasure.hellinger2(capital_gain, capital_gain_transformed))
    print('capital_loss: ', DistanceMeasure.hellinger2(capital_loss, capital_loss_transformed))
    print('hours_per_week: ', DistanceMeasure.hellinger2(hours_per_week, hours_per_week_transformed))


