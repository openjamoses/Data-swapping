import numpy as np
import tensorflow as tf
import os,time
import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot
from sklearn.preprocessing import KBinsDiscretizer

from src.models.v3.load_data import LoadData


def d(x,y):
    return (tf.expand_dims(x,1)-tf.expand_dims(y,0))**2
def discretization(x, N):
    # UNIFORM DISCRETIZATION OF ran(X)
    Xhat = tf.constant(np.linspace(0 ,1 ,N))   # => q below is a dist. over xhat=[0,.01,...,.99,1]
    layer = tf.keras.layers.Discretization(num_bins=2, epsilon=0.01)
    layer.adapt(x)
    print(Xhat)
    print('layer(input): ', layer(x))
    return Xhat


if __name__ == '__main__':
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [3, 2, 0, 4, 5, 1, 7, 6, 2, 1]
    Xhat = discretization(y, len(y))

    path = '../../../dataset/'
    correlation_threshold = 0.45
    loadData = LoadData(path, threshold=correlation_threshold)  # ,threshold=correlation_threshold
    data_name = 'adult_'  # _35_threshold

    #data = loadData.load_adult_data('adult.data.csv')
    data = loadData.load_compas_data('compas-scores-two-years.csv')

    print(data.head())

    #discrete_x = d(y,Xhat)
    #print(discrete_x)

    # discretization transform the raw data
    y = np.array(y).reshape((len(y), 1))
    #kbins = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform')
    kbins = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='quantile')
    #data_trans = kbins.fit_transform(data)
    #print(data_trans)

    #data['age'] = kbins.fit_transform(data['age'].values.reshape(-1, 1))
    #data['hours-per-week'] = kbins.fit_transform(data['hours-per-week'].values.reshape(-1, 1)) #pd.qcut(data['hours-per-week'], q=2)
    #data['capital-gain'] = kbins.fit_transform(data['capital-gain'].values.reshape(-1, 1)) #pd.qcut(data['capital-gain'], q=2, duplicates='drop')
    #data['capital-loss'] = kbins.fit_transform(data['capital-loss'].values.reshape(-1, 1)) # pd.qcut(data['capital-loss'], q=2, duplicates='drop')

    #print(data)
    # histograms of the variables
    #print(data)
    #data.hist()
    #pyplot.show()

    #print(dataset)


