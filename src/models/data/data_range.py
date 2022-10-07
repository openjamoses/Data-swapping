import random

import numpy as np


def get_categorical_features(data,feature_index):
    print('column data: ', set(data[0:, feature_index]))
    return list(set(data[0:, feature_index]))


def get_categorical_features_posible_value(data, feature_index):
    print('column data: ', set(data[0:, feature_index]))
    data_folded = {}
    for val in list(set(data[0:, feature_index])):
        data_folded[val] = val
    return data_folded  # list(set(self.data[0:, feature_index]))

def determine_range(data, feature_index, k_folds=3):
    data_range = data[0:, feature_index]
    # interval_ = round(((max(sorted_)-min(sorted_))/k_folds),0)
    fold_count = 0
    folded_data = {}
    # percentile_25 = np.percentile(data_range, 25)
    percentile_50 = np.percentile(data_range, 50)
    percentile_100 = np.percentile(data_range, 100)
    if percentile_50 == np.min(data_range) or percentile_50 == np.max(data_range):
        # percentile_25 = np.percentile(np.unique(data_range), 25)
        percentile_50 = np.percentile(np.unique(data_range), 50)
        # percentile_75 = np.percentile(np.unique(data_range), 75)
    # if percentile_50 == percentile_25:
    #    percentile_50 = np.max(data_range)/2
    # if percentile_25 == np.min(data_range):
    #    percentile_25 = percentile_50/2
    for i in range(len(data_range)):
        fold_id = percentile_50
        if data_range[i] <= percentile_50:
            fold_id = percentile_50
        elif data_range[i] > percentile_50:  # and data_range[i] <= percentile_50:
            fold_id = percentile_100
        # elif data_range[i] > percentile_50:
        #    fold_id = percentile_75
        if fold_id in folded_data.keys():
            folded_data[fold_id].add(data_range[i])
        else:
            folded_data[fold_id] = set([data_range[i]])
    for key, val in folded_data.items():
        if len(val) < 3:
            print('Category <3: ', np.min(list(val)), percentile_50, percentile_100)
            if key == percentile_50:
                val2 = []
                val2.append(random.uniform(np.min(list(val)), np.max(list(val))))
                val2.append(random.uniform(np.min(list(val)), np.max(list(val))))
                val2.extend(list(val))
            if key == percentile_50:
                val2 = []
                # print('percentile_50: ', percentile_50, val)
                # val2.append(random.uniform(percentile_50, np.min(list(val))))
                # val2.append(random.uniform(percentile_50, np.min(list(val))))
                val2.append(random.uniform(percentile_50, np.max(list(val))))
                val2.append(random.uniform(percentile_50, np.max(list(val))))
                val2.extend(list(val))
            '''if key == percentile_75:
                val2 = []
                val2.append(random.uniform(percentile_50, percentile_100))
                val2.append(random.uniform(percentile_50, percentile_100))
                val2.extend(list(val))'''
            folded_data[key] = list(val)
        else:
            folded_data[key] = list(val)
    return folded_data  # list(folded_data.values())
def check_member_continous(folded_data, value):
    k_return = ''
    for k, vL in folded_data.items():
        if value in vL:
            k_return = k
            break
    return k_return