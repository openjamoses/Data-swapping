import operator

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import random

def split_features_target(data, index=20):
    #print(data[0:, 0:index], data[0:, index:])
    x= np.concatenate((data[0:, 0:index], data[0:, index+1:]), axis=1)
    print(x.shape)
    return x, data[0:, index]
    #return data[0:, 0:index]+data[0:, index:], data[0:, index]
def data_split(data, sample_size=0.25):
    return train_test_split(data, train_size=None, test_size=sample_size, random_state=42)
def scores_metrics(gda_pred, y_test):
    accuracy = round(metrics.accuracy_score(y_test, gda_pred),3)          # accuracy: (tp + tn) / (p + n)
    precision = round(metrics.precision_score(y_test, gda_pred),3)          # precision tp / (tp + fp)
    recall = round(metrics.recall_score(y_test, gda_pred) ,3)               # recall: tp / (tp + fn)
    f1 = round(metrics.f1_score(y_test, gda_pred) ,3)                       # f1: 2 tp / (2 tp + fp + fn)

    print('precision: ', precision, 'recall: ', recall, 'f1: ', f1,  'accuracy: ', accuracy)
    return precision, recall, f1, accuracy

def sort_dict(dict_, reverse=True):
    return dict(sorted(dict_.items(), key=operator.itemgetter(1), reverse=reverse))

def subclass_probablity(x_test, y_test, y_pred, sensitive_index):
    data = {}
    for i in range(len(x_test)):
        if x_test[i][sensitive_index] in data.keys():
            if y_pred[i] in data[x_test[i][sensitive_index]].keys():
                if y_test[i] in data[x_test[i][sensitive_index]][y_pred[i]].keys():
                    data[x_test[i][sensitive_index]][y_pred[i]][y_test[i]] += 1
                else:
                    data[x_test[i][sensitive_index]][y_pred[i]][y_test[i]] = 1
            else:
                data[x_test[i][sensitive_index]][y_pred[i]] = {}
                data[x_test[i][sensitive_index]][y_pred[i]][y_test[i]] = 1
        else:
            data[x_test[i][sensitive_index]] = {}
            data[x_test[i][sensitive_index]][y_pred[i]] = {}
            data[x_test[i][sensitive_index]][y_pred[i]][y_test[i]] = 1

        #data[x_test[i][sensitive_index]][y_pred[i]][y_test[i]] = data.get([x_test[i][sensitive_index]][y_pred[i]][y_test[i]], 0) + 1
    return data

def transform_df_minmax(df_data):
    scaler = MinMaxScaler()
    df_data = pd.DataFrame(scaler.fit_transform(df_data), columns=df_data.columns)
    #dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.2, shuffle=True)
    return df_data

def fairness_metrics(x_test, y_test, y_pred, sensitive_index, protected=0, unprotected=1):
    data = subclass_probablity(x_test,y_test,y_pred,sensitive_index)
    AOD = calculate_average_odds_difference(data, protected=protected,unprotected=unprotected)
    DI = calculate_Disparate_Impact(data, protected=protected, unprotected=unprotected)
    DIR = calculate_Disparate_Impact_ratio(data, protected=protected, unprotected=unprotected)
    SPD = calculate_SPD(data, protected=protected, unprotected=unprotected)
    EOD = calculate_equal_opportunity_difference(data, protected=protected, unprotected=unprotected)
    TPR = calculate_TPR_difference(data, protected=protected, unprotected=unprotected)
    FPR = calculate_FPR_difference(data, protected=protected, unprotected=unprotected)
    print('AOD: ', AOD, 'DI: ', DI, 'DIR: ', DIR, 'SPD: ', SPD, 'EOD: ', EOD, 'TPR: ', TPR, 'FPR: ', FPR)
    return AOD, DI, DIR, SPD, EOD, TPR, FPR

def calculate_average_odds_difference(data , protected=0, unprotected=1):
    # TPR_male = TP_male/(TP_male+FN_male)
    # TPR_female = TP_female/(TP_female+FN_female)
    # FPR_male = FP_male/(FP_male+TN_male)
    # FPR_female = FP_female/(FP_female+TN_female)
    # average_odds_difference = abs(abs(TPR_male - TPR_female) + abs(FPR_male - FPR_female))/2
    #FPR_diff = calculate_FPR_difference(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female)
    #TPR_diff = calculate_TPR_difference(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female)
    FPR_diff = calculate_FPR_difference(data=data,protected=protected, unprotected=unprotected)
    TPR_diff = calculate_TPR_difference(data=data,protected=protected, unprotected=unprotected)

    average_odds_difference = (FPR_diff + TPR_diff)/2
    #print("average_odds_difference",average_odds_difference)
    return round(average_odds_difference,3)
def extract_stats(data , protected=0, unprotected=1):
    TP_1 = 0
    FP_1 = 0
    TN_1 = 0
    FN_1 = 0

    FN_0 = 0
    FP_0 = 0
    TP_0 = 0
    TN_0 = 0
    if unprotected in data.keys():
        if 0 in data[unprotected].keys():
            if 0 in data[unprotected][0].keys():
                TN_1 = data[unprotected][0][0]
            if 1 in data[unprotected][0].keys():
                FN_1 = data[unprotected][0][1]
        if 1 in data[unprotected].keys():
            if 1 in data[unprotected][1].keys():
                TP_1 = data[unprotected][1][1]
            if 0 in data[unprotected][1].keys():
                FP_1 = data[unprotected][1][0]
    if protected in data.keys():
        if 0 in data[protected].keys():
            if 0 in data[protected][0].keys():
                TN_0 = data[protected][0][0]
            if 1 in data[protected][0].keys():
                FN_0 = data[protected][0][1]
        if 1 in data[protected].keys():
            if 1 in data[protected][1].keys():
                TP_0 = data[protected][1][1]
            if 0 in data[protected][1].keys():
                FP_0 = data[protected][1][0]

    return TP_1, FP_1, TN_1, FN_1, TP_0, FP_0, TN_0, FN_0

def calculate_Disparate_Impact(data , protected=0, unprotected=1):
    TP_1, FP_1, TN_1, FN_1, TP_0, FP_0, TN_0, FN_0 = extract_stats(data,protected,unprotected)
    #P1 = (data.get(unprotected,0).get(1,0)+data[unprotected][0][1])/(data[unprotected][1][1]+data[unprotected][0][0]+
    #                                                        data[unprotected][1][0]+data[unprotected][0][1])
    #P0 = (data[protected][1][1] + data[protected][0][1]) / (data[protected][1][1] + data[protected][0][0] +
    #                                                       data[protected][1][0] + data[protected][0][1])


    P1 = (TP_1 + FP_1)/(TP_1 + TN_1 + FN_1 + FP_1)
    P0 = (TP_0 + FP_0)/(TP_0 + TN_0 + FN_0 +  FP_0)
    DI = (P0/P1)
    return round((1 - abs(DI)),3)

def calculate_Disparate_Impact_ratio(data , protected=0, unprotected=1):

    P1 = data[unprotected][1][1]/np.sum(list(data[unprotected][1].values()))
    P0 = data[protected][1][1] / np.sum(list(data[protected][1].values()))
    #P_male = (TP_male + FP_male)/(TP_male + TN_male + FN_male + FP_male)
    #P_female =  (TP_female + FP_female)/(TP_female + TN_female + FN_female +  FP_female)
    DI = (P0/P1)
    return round(DI,3)

def calculate_SPD(data , protected=0, unprotected=1):
    #P_male = (TP_male + FP_male)/(TP_male + TN_male + FN_male + FP_male)
    #P_female = (TP_female + FP_female) /(TP_female + TN_female + FN_female +  FP_female)
    TP_1, FP_1, TN_1, FN_1, TP_0, FP_0, TN_0, FN_0 = extract_stats(data, protected, unprotected)
    #P1 = (data[unprotected][1][1] + data[unprotected][0][1]) / (data[unprotected][1][1] + data[unprotected][0][0] +
    #                                                            data[unprotected][1][0] + data[unprotected][0][1])
    #P0 = (data[protected][1][1] + data[protected][0][1]) / (data[protected][1][1] + data[protected][0][0] +
    #                                                        data[protected][1][0] + data[protected][0][1])
    P1 = (TP_1 + FP_1)/(TP_1 + TN_1 + FN_1 + FP_1)
    P0 = (TP_0 + FP_0) /(TP_0 + TN_0 + FN_0 +  FP_0)
    SPD = (P0 - P1)
    return round(abs(SPD),3)
def calculate_TPR(TP, FP, TN,FN):
    #TPR = TP/(TP+FN)
    TPR = 0
    if (TP+FN) > 0:
        TPR = TP/(TP+FN)
    return round(TPR,3)
def calculate_FPR(TP, FP, TN,FN):
    #FPR = FP/(FP+TN)
    FPR = 0
    if (FP+TN) > 0:
        FPR = FP/(FP+TN)
    return round(FPR,3)

def calculate_SP(TP, FP, TN,FN):
    SP = 0
    if (TP + TN + FN + FP) > 0:
        SP = (TP + FP) / (TP + TN + FN + FP)
    return round(SP,3)
def calculate_proportion(TP, FP, TN,FN):
    Probability = 0
    if (TP + TN + FN + FP) > 0:
        Probability = (TP + FP) / (TP + TN + FN + FP)
    return round(Probability,3)
def calculate_equal_opportunity_difference(data , protected=0, unprotected=1):
    # TPR_male = TP_male/(TP_male+FN_male)
    # TPR_female = TP_female/(TP_female+FN_female)
    # equal_opportunity_difference = abs(TPR_male - TPR_female)
    #print("equal_opportunity_difference:",equal_opportunity_difference)
    return calculate_TPR_difference(data,protected=protected, unprotected=unprotected)
    #return calculate_TPR_difference(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female)

def calculate_TPR_difference(data , protected=0, unprotected=1):
    #TPR_male = TP_male/(TP_male+FN_male)
    #TPR_female = TP_female/(TP_female+FN_female)
    TPR_male = data[unprotected][1][1] / (data[unprotected][1][1] + data[unprotected][1][0])
    TPR_female = data[protected][1][1] / (data[protected][1][1] + data[protected][1][0])
    # print("TPR_male:",TPR_male,"TPR_female:",TPR_female)
    diff = (TPR_male - TPR_female)
    return round(diff,3)

def calculate_FPR_difference(data , protected=0, unprotected=1):
    #FPR_male = FP_male/(FP_male+TN_male)
    #FPR_female = FP_female/(FP_female+TN_female)
    TP_1, FP_1, TN_1, FN_1, TP_0, FP_0, TN_0, FN_0 = extract_stats(data, protected, unprotected)

    FPR_1 = FP_1 / (FP_1 + TN_1)
    FPR_0 = FP_0 / (FP_0 + TN_0)
    # print("FPR_male:",FPR_male,"FPR_female:",FPR_female)
    diff = (FPR_0 - FPR_1)
    return round(diff,3)
