'''
Group discrimination testing for Subject System A for Credit dataset
Inputs :
    argv[1] : Train file
'''
from __future__ import division
from random import seed, shuffle
import random
import math
import os
import numpy as np
import matplotlib.pyplot as plt # for plotting stuff
from random import seed, shuffle
from scipy.stats import multivariate_normal # generating synthetic data

from collections import defaultdict
from sklearn import svm
import os, sys
#import urllib2

sys.path.insert(0, '../../examples/fair_classification/')  # the code for fair classification is in this directory
import utils as ut
import numpy as np
import pandas as pd
import itertools
import loss_funcs as lf  # loss funcs that can be optimized subject to various constraints
#import commands

# Minimum number of inputs tos tart applying the confidence check optimization
minInp = 15000

# Maximum number of inputs to test against
max_inp = 15000

# set prinsuite to 0 to not print the test suite
printsuite = 1

# Training file
trainfile = '../dataset/Credit' #sys.argv[1]

random.seed(2)
# fixed seed to get the same test suite each time

sens_arg = 9
fixval = 9

if (sens_arg == 9):
    name = 'sex'
    cov = 0
else:
    name = 'race'
    cov = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

X = []
Y = []
i = 0
sensitive = {};
sens = []
outputfile = "../Suites/freshAcredit" + str(sens_arg) + str(fixval) + ".txt"
option = 4
already = "../Suites/Acredit"
path = '../dataset/'
df_credit = pd.read_excel(open(path+'Dataset.xlsx', 'rb'), sheet_name='Credit', engine="openpyxl")

list_mean = {}
#n_samples = 1000  # generate these many data points per class
disc_factor = math.pi / 4.0  # this variable determines the initial discrimination in the data -- decraese it to generate more discrimination


def gen_gaussian(mean_in, cov_in, class_label, n_samples=10000):
    nv = multivariate_normal(mean=mean_in, cov=cov_in)
    X = nv.rvs(n_samples)
    y = np.ones(n_samples, dtype=float) * class_label
    return nv, X, y



data_mean = {}
data_count = {}
for column in df_credit.columns.tolist():
    if column != 'i' and column != 'j':
        for i in range(len(df_credit) - 1):
            if df_credit.u.values[i] in data_mean.keys():
                data_count[df_credit.u.values[i]] += 1
                if column in data_mean[df_credit.u.values[i]].keys():
                    data_mean[df_credit.u.values[i]][column].append(df_credit[column][i])
                else:
                    data_mean[df_credit.u.values[i]][column] = [df_credit[column][i]]
            else:
                data_mean[df_credit.u.values[i]] = {}
                data_mean[df_credit.u.values[i]][column] = [df_credit[column][i]]
                data_count[df_credit.u.values[i]] = 1


print(data_mean)
data_mean_val = {}
data_cov_val = {}
sensive_columns = ['i', 'j']
for key, val in data_mean.items():
    list_ = []
    for key2, val2 in val.items():
        if not key2 in sensive_columns:
            list_.append(val2)
            if key in data_mean_val.keys():
                data_mean_val[key].append(np.mean(val2))
            else:
                data_mean_val[key] = [np.mean(val2)]

    #data_cov_val[key] = np.cov(np.array(list(val.values())),bias=True)
    data_cov_val[key] = np.cov(np.array(list_), bias=True)
print(data_mean_val)
print(data_cov_val)

""" Generate the non-sensitive features randomly """
# We will generate one gaussian cluster for each class
#mu1, sigma1 = [2, 2], [[5, 1], [1, 5]]
#mu2, sigma2 = [-2, -2], [[10, 1], [1, 3]]

print(data_mean_val[1])
nv1, X1, y1 = gen_gaussian(np.array(data_mean_val[1]), np.array(data_cov_val[1]), 1,data_count[1])  # positive class
nv2, X2, y2 = gen_gaussian(data_mean_val[0], data_cov_val[0], 0, data_count[1])  # negative class

# join the posisitve and negative class clusters
X = np.vstack((X1, X2))
y = np.hstack((y1, y2))

# shuffle the data
perm = list(range(0, data_count[1]+data_count[0]))
shuffle(perm)
X = X[perm]
y = y[perm]

rotation_mult = np.array(
    [[math.cos(disc_factor), -math.sin(disc_factor)], [math.sin(disc_factor), math.cos(disc_factor)]])
X_aux = np.dot(X, rotation_mult)
x_control = []
for i in range(len(X)):
    x = X_aux[i]

    # probability for each cluster that the point belongs to it
    p1 = nv1.pdf(x)
    p2 = nv2.pdf(x)

    # normalize the probabilities from 0 to 1
    s = p1 + p2
    p1 = p1 / s
    p2 = p2 / s

    r = np.random.uniform()  # generate a random number from 0 to 1

    if r < p1:  # the first cluster is the positive class
        x_control.append(1.0)  # 1.0 means its male
    else:
        x_control.append(0.0)  # 0.0 -> female
x_control = np.array(x_control)

num_to_draw = 200  # we will only draw a small number of points to avoid clutter
x_draw = X[:num_to_draw]
y_draw = y[:num_to_draw]
x_control_draw = x_control[:num_to_draw]

X_s_0 = x_draw[x_control_draw == 0.0]
X_s_1 = x_draw[x_control_draw == 1.0]
y_s_0 = y_draw[x_control_draw == 0.0]
y_s_1 = y_draw[x_control_draw == 1.0]
plt.scatter(X_s_0[y_s_0 == 1.0][:, 0], X_s_0[y_s_0 == 1.0][:, 1], color='green', marker='x', s=30, linewidth=1.5,
            label="Prot. +ve")
plt.scatter(X_s_0[y_s_0 == -1.0][:, 0], X_s_0[y_s_0 == -1.0][:, 1], color='red', marker='x', s=30, linewidth=1.5,
            label="Prot. -ve")
plt.scatter(X_s_1[y_s_1 == 1.0][:, 0], X_s_1[y_s_1 == 1.0][:, 1], color='green', marker='o', facecolors='none', s=30,
            label="Non-prot. +ve")
plt.scatter(X_s_1[y_s_1 == -1.0][:, 0], X_s_1[y_s_1 == -1.0][:, 1], color='red', marker='o', facecolors='none', s=30,
            label="Non-prot. -ve")

plt.tick_params(axis='x', which='both', bottom='off', top='off',
                labelbottom='off')  # dont need the ticks to see the data distribution
plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
plt.legend(loc=2, fontsize=15)
plt.xlim((-15, 10))
plt.ylim((-10, 15))
plt.savefig("img/data.png")
plt.show()


with open(trainfile, "r") as ins:
    for line in ins:
        line = line.strip()
        line1 = line.split(',')
        if (i == 0):
            i += 1
            continue
        #L = map(int, line1[:-1])
        L = [int(i) for i in line1[:-1]]
        sens.append(L[sens_arg - 1])
        # L[sens_arg-1]=-1
        X.append(L)

        if (int(line1[-1]) == 0):
            Y.append(-1)
        else:
            Y.append(1)

X = np.array(X, dtype=float);
Y = np.array(Y, dtype=float);
sensitive[name] = np.array(sens, dtype=float);
loss_function = lf._logistic_loss;
sep_constraint = 0;
sensitive_attrs = [name];
sensitive_attrs_to_cov_thresh = {name: cov};