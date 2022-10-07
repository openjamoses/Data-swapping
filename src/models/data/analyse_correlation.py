from numpy import mean, cov
from numpy import std
import pandas as pd
from numpy.random import randn
from numpy.random import seed
from matplotlib import pyplot
# seed random number generator
from src.models.data.convolution_ import Fourier_transform
from scipy.stats import pearsonr, spearmanr

seed(1)
# prepare data
data1 = 20 * randn(1000) + 100
data2 = data1 + (10 * randn(1000) + 50)



# summarize
print('data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))
print('data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))
# plot


path = '../dataset/'
#df = pd.read_csv(path+'logging/log_KL_Divergence_overral_adult-45.csv')
df_global = pd.read_csv(path + 'logging2/log_KL_Divergence_overral_adult-45_2_50.csv')
ranking_attributes = ['JS_Divergence', 'Casuality',  'Importance'] # , 'SP', 'Casuality',
#ranking_attributes = ['JS_Divergence', 'Casuality']  # , 'SP', 'Casuality',

JS_Divergence = df_global.JS_Divergence.values
Casuality = df_global.Casuality.values
Importance = df_global.Importance.values

#DFT_JS_Divergence = Fourier_transform.DFT(JS_Divergence)
#print("DFT_JS_Divergence: ", DFT_JS_Divergence)
#DFT_Casuality = Fourier_transform.DFT(Casuality)
#print("DFT_Casuality: ", DFT_Casuality)
print('JS_Divergence before: ', JS_Divergence)
print('JS_Divergence before: ', JS_Divergence*0.5)
pyplot.scatter(JS_Divergence, Importance)
pyplot.show()

# calculate covariance matrix
covariance = cov(data1, data2)
print(covariance)
# calculate Pearson's correlation
corr, _ = pearsonr(JS_Divergence, Importance)
print('Pearsons correlation: %.3f' % corr, _)
print('Correlation manual: ', covariance/(std(JS_Divergence)*std(Importance)))

corr, _ = spearmanr(data1, data2)
print('Spearmans correlation: %.3f' % corr, _)