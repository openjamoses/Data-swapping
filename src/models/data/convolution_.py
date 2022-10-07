import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import scipy
import math
from scipy import integrate, stats

def modify_df(x, modifications):
	return x + modifications

class Fourier_transform:
    # Graphing helper function
    @staticmethod
    def setup_graph(title='', x_label='', y_label='', fig_size=None):
        fig = plt.figure()
        if fig_size != None:
            fig.set_size_inches(fig_size[0], fig_size[1])
        ax = fig.add_subplot(111)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    @staticmethod
    def DFT(x):
        """
            Compute the discrete Fourier Transform of the 1D array x
            :param x: (array)
            """

        N = x.size
        n = np.arange(N)
        k = n.reshape((N, 1))
        e = np.exp(-2j * np.pi * k * n / N)
        return np.dot(e, x)
    @staticmethod
    def FFT(x):
        fft_output = np.fft.fft(x)
        #magnitude_only = [np.sqrt(i.real ** 2 + i.imag ** 2) / len(fft_output) for i in fft_output]
        #frequencies = [(i * 1.0 / num_samples) * sample_rate for i in range(num_samples // 2 + 1)]
        #print(fft_output)
        return fft_output
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
        #distance = (1/math.sqrt(2) )* integrate((math.sqrt(Y1)-math.sqrt(Y2))**2)
        #d1 = scipy.stats.entropy(Y1, Y2, base=2)
        print([(math.sqrt(p_i) - math.sqrt(q_i)) ** 2 for p_i, q_i in zip(Y1, Y2)])
        return math.sqrt(integrate([(math.sqrt(p_i) - math.sqrt(q_i)) ** 2 for p_i, q_i in zip(Y1, Y2)]) / 2)

        #return distance
    @staticmethod
    def hellinger2(p, q):
        """Hellinger distance between two discrete distributions.
           In pure Python.
           Some improvements.
        """
        return math.sqrt(sum([(math.sqrt(p_i) - math.sqrt(q_i)) ** 2 for p_i, q_i in zip(p, q)]) / 2)

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
        return min(max(dist, 0.), 1.)


path = '../dataset/'
if __name__ == '__main__':
    #df = pd.read_csv(path+'logging/log_KL_Divergence_overral_adult-45.csv')
    df_global = pd.read_csv(path + 'logging2/log_KL_Divergence_overral_adult-45_2_50.csv')
    ranking_attributes = ['JS_Divergence', 'Casuality',  'Importance'] # , 'SP', 'Casuality',
    #ranking_attributes = ['JS_Divergence', 'Casuality']  # , 'SP', 'Casuality',

    JS_Divergence = df_global.JS_Divergence.values
    Casuality = df_global.Casuality.values
    Importance = df_global.Importance.values

    print('numpy: ',df_global.to_numpy())

    #DFT_JS_Divergence = Fourier_transform.DFT(JS_Divergence)
    #print("DFT_JS_Divergence: ", DFT_JS_Divergence)
    #DFT_Casuality = Fourier_transform.DFT(Casuality)
    #print("DFT_Casuality: ", DFT_Casuality)



    print(Fourier_transform.hellinger_dist(JS_Divergence, JS_Divergence*0.55))
    print(Fourier_transform.hellinger2(JS_Divergence, JS_Divergence * 0.55))
    #print("\n************\n")

    #DFT_JS_Divergence = Fourier_transform.FFT(JS_Divergence)
    #print("DFT_JS_Divergence2: ", DFT_JS_Divergence)
    #DFT_Casuality = Fourier_transform.FFT(Casuality)
    #print("DFT_Casuality2: ", DFT_Casuality)
    #PSA = 'Feature'
    #sub_category = 'Category'
    #ranking = Ranking()
    #rank_global = ranking.rank_average(df_global,feature_name=PSA, PSA=ranking_attributes)
    #rank_global = ranking.sort_dict(rank_global, reverse=False)
    #print('Global ranking: ', rank_global)

    '''
    # examples
    freq = 1  # hz - cycles per second
    amplitude = 3
    time_to_plot = 2  # second
    sample_rate = 100  # samples per second
    num_samples = sample_rate * time_to_plot

    t = np.linspace(0, time_to_plot, num_samples)
    signal = [amplitude * np.sin(freq * i * 2 * np.pi) for i in t]  # Explain the 2*pi

    print('t: ', t)
    print('signal: ', signal)





    ### KL divergence
    num_samples = len(JS_Divergence)
    time_slots = range(len(JS_Divergence))
    fft_output = np.fft.rfft(JS_Divergence)

    print('fft_output: ', fft_output)
    magnitude_only = [np.sqrt(i.real ** 2 + i.imag ** 2) / len(fft_output) for i in fft_output]
    frequencies = [(i * 1.0 / num_samples) * sample_rate for i in range(num_samples // 2 + 1)]
    print('frequencies: ', frequencies)
    print('magnitude_only: ', magnitude_only)
    Fourier_transform.setup_graph(x_label='time (in seconds)', y_label='amplitude', title='time domain')
    plt.plot(time_slots, JS_Divergence)
    plt.show()

    Fourier_transform.setup_graph(x_label='frequency (in Hz)', y_label='amplitude', title='frequency domain')
    plt.plot(frequencies, magnitude_only, 'r')
    plt.show()'''


