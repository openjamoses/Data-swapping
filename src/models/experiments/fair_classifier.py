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
from collections import defaultdict
from sklearn import svm
import os, sys
#import urllib2

sys.path.insert(0, '../../examples/fair_classification/')  # the code for fair classification is in this directory
import utils as ut
import numpy as np
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

gamma = None

print(sensitive)
w = ut.train_model(X, Y, sensitive, loss_function, 1, 0, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh,
                   gamma);
#print(w)

list_data = ['Acredit',  'Acausalcredit']
for data in list_data:
    test_suits = '../suits/'+data
    f = open(test_suits,"r")
    done = {}

    pos, neg = 0,0

    pos1,pos0, neg1, neg0 = 0,0,0,0
    for line in f:
        line =  line.strip()
        #line_split  =  line[:line.rindex(',')].split(',')
        line_split = line.split(',')
        #line = ','.join(line[:-1])
        #line+=','
        #done[line]=1
        x = [int(i) for i in line_split[:-1]]
        y = float(line[line.rindex(',')+1:].replace(' ', ''))
        #y2 = []
        #y_o
        #for i in y:
        #    if i[0] == '-':
        #        y2.append(0)
        #    else:
        #        y2.append(1)

        #print(np.array(x), np.array(x).shape)
        #print(y, np.sign(np.dot(w,x)))
        if y == 1:
            pos += 1
            if x[8] == 1:
                pos1 += 1
            else:
                pos0 += 1
        else:
            neg += 1
            if x[8] == 1:
                neg1 += 1
            else:
                neg0 += 1
    frac=pos*1.0/(pos+neg)
    score_1 = 2.5*math.sqrt(frac*(1-frac)*1.0/(pos+neg))
    total_  = pos+neg
    score = 100*(pos1*1.0/(pos1+neg1) - pos0*1.0/(pos0+neg0))
    print(frac, score_1, score, total_ )