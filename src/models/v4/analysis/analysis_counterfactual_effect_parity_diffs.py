import csv

import pandas as pd
import numpy as np
import os

# todo: group by tutorials: https://stackoverflow.com/questions/17679089/pandas-dataframe-groupby-two-columns-and-get-counts
#from src.models.v3.load_data import LoadData
from src.models.v3.utility_functions import data_split, split_features_target
from src.models.v4.load_data_original import LoadData
from src.models.v4.sensitivity_utils_4_3 import is_categorical

alpha = 0.3
path = '../../dataset/'
# if __name__ == '__main__':
# df = pd.read_csv(path+'logging/log_KL_Divergence_overral_adult-45.csv')
# data_name = 'adult'
# data_name = 'clevelan_heart'
# data_name = 'Student-{}_'.format(alpha)  # _35_threshold
# data_name = 'Student-quantile-{}_'.format(alpha)  # _35_threshold
# data_name = 'german_credit-{}_'.format(alpha)  # _35_threshold
data_name = 'compas-{}_'.format(alpha)  # _35_threshold
# data_name = 'Student', clevelan_heart
path_output = 'logging3/{}/'.format(data_name)
#df_adult = loadData.load_compas_data('compas-scores-two-years.csv')
#df_adult = loadData.load_student_data('Student.csv')
#df_adult = loadData.load_clevelan_heart_data('processed.cleveland.data.csv')
correlation_threshold = 0.45  # 0.35
loadData = LoadData(path, threshold=correlation_threshold)  # ,threshold=correlation_threshold
#df_adult, df_original = loadData.load_bank_data('bank.csv')
#df_adult, df_original = loadData.load_compas_data('compas-scores-two-years.csv')
#df_adult, df_original = loadData.load_clevelan_heart_data('processed.cleveland.data.csv')
df_adult, df_original = loadData.load_compas_data('compas-scores-two-years.csv')
colums_list = df_adult.columns.tolist()

#print(df_original)
#print(df_adult)
target_name = loadData.target_name
target_index = loadData.target_index
data = df_adult.to_numpy()
data_org = df_original.to_numpy()
data_indices = [i for i in range(len(data))]
train_indices, test_indices = data_split(data=data_indices, sample_size=0.25)
train, test = data[train_indices], data[test_indices]
x_train, y_train = split_features_target(train, index=target_index)
x_test, y_test = split_features_target(test, index=target_index)

def is_integer(val):
    if float(val).is_integer():
        return int(float(val))
    else:
        return round(float(val),1)
#todo: select by indices
test_org = data_org[test_indices]
data_dict_ = {}
'''data_dict_['housing'] = {}
data_dict_['housing'][1] = 'Yes'
data_dict_['housing'][0] = 'No'
data_dict_['default'] = {}
data_dict_['default'][0] = 'Yes'
data_dict_['default'][1] = 'No'
data_dict_['loan'] = {}
data_dict_['loan'][0] = 'Yes'
data_dict_['loan'][1] = 'No'

data_dict_['education'] = {}
data_dict_['education'][1] = 'primary'
data_dict_['education'][2] = 'secondary'
data_dict_['education'][3] = '[secondary, tertiary]'
data_dict_['education'][0] = 'primary, unknown'
'''


### compas

data_dict_['sex'] = {}
data_dict_['sex'][1] = 'Female'
data_dict_['sex'][0] = 'Male'
data_dict_['race'] = {}
data_dict_['race'][0] = 'Caucasian'
data_dict_['race'][1] = 'None-Caucasian'

data_dict_['priors_count'] = {}
data_dict_['priors_count'][1] = '1-3'
data_dict_['priors_count'][0] = '>3'

data_dict_['c_charge_degree'] = {}
data_dict_['c_charge_degree'][0] = 'M'
data_dict_['c_charge_degree'][1] = 'F'

#Cleveran Heart data
'''
data_dict_['age'] = {}
#age_l = df_original.df_adult.values.tolist()
#age_org_l = df_original.age.values.tolist()

mean = df_original.loc[:, "age"].mean()
#for i in range(len())
data_dict_['age'][0] = '>={}'.format(round(mean))
data_dict_['age'][1] = '<{}'.format(round(mean))
data_dict_['sex'] = {}
data_dict_['sex'][0] = 'Female'
data_dict_['sex'][1] = 'Male'
data_dict_['exang'] = {}
data_dict_['exang'][0] = 'No'
data_dict_['exang'][1] = 'Yes'

data_dict_['fbs'] = {}
data_dict_['fbs'][0] = 'No'
data_dict_['fbs'][1] = 'Yes'

data_dict_['restecg'] = {}
data_dict_['restecg'][0] = 'normal'
data_dict_['restecg'][1] = 'ST-T\n wave'
data_dict_['restecg'][2] = 'probable'

data_dict_['thal'] = {}
data_dict_['thal'][3] = 'normal'
data_dict_['thal'][6] = 'fixed\n defect'
data_dict_['thal'][7] = 'reversable\n defect'
'''

def determin_parity(feature_name, category):
    column_id = colums_list.index(feature_name)
    data_counts = {}
    cat1_count = 0
    cat2_count = 0
    #if feature_name == 'education':
    #    print()
    C1, C2 = category[0], category[1]
    data_category = {}
    if is_categorical(x_test, column_id):
        # todo: first feature is discrete

        if category[0] > category[1]:
            C1, C2 = category[1], category[0]
        #print('category[0]: ', category[0], category[1], list(set(data[0:, column_id])), feature_name)
        print(feature_name, data_dict_[feature_name], category)
        if feature_name in data_dict_.keys():
            data_category['cat1'] = data_dict_[feature_name][is_integer(C1)]
            data_category['cat2'] = data_dict_[feature_name][is_integer(C2)]
        else:
            data_category['cat1'] = is_integer(C1)
            data_category['cat2'] = is_integer(C2)
        for i in range(len(x_test)):
            if x_test[i][column_id] == C1:
                #data_counts[category[0]] = data_counts.get(category[0], 0) + 1
                cat1_count += 1
            else:
                #data_counts[category[1]] = data_counts.get(category[1], 0) + 1
                cat2_count += 1
    else:
        if '>' in str(C1):
            C1, C2 = category[1], category[0]

        rm_cat_1 = C1.replace('>=','')
        rm_cat_1 = float(rm_cat_1.replace('<=', ''))

        rm_cat_2 = C2.replace('>=', '')
        rm_cat_2 = float(rm_cat_2.replace('<=', ''))

        data_ = {}
        for i in range(len(x_test)):
            #print(rm_cat_1, x_test[i][column_id])
            if x_test[i][column_id] <= rm_cat_1:
                #data_counts[rm_cat_1] = data_counts.get(rm_cat_1, 0) + 1
                cat1_count += 1
                cat_ = 'cat1'
                if cat_ in data_.keys():
                    data_[cat_].append(test_org[i][column_id])
                else:
                    data_[cat_] = [test_org[i][column_id]]
            else:
                #data_counts[rm_cat_2] = data_counts.get(rm_cat_2, 0) + 1
                cat2_count += 1
                cat_ = 'cat2'
                #print(x_test[i][column_id])
                if cat_ in data_.keys():
                    data_[cat_].append(test_org[i][column_id])
                else:
                    data_[cat_] = [test_org[i][column_id]]
        #cat_1 = int(max(data_['cat1']))
        #cat_1 = int(max(data_['cat1']))
        #print(x_test)
        #print(feature_name, C1, C2, rm_cat_1, rm_cat_2, category, data_)

        if len(data_['cat1']) <5:
            cat_1 = data_['cat1']
            cat_2 = data_['cat2']
        else:
            cat_1 = '<={}'.format(is_integer(max(data_['cat1'])))
            cat_2 = '>{}'.format(is_integer(max(data_['cat1'])))

        data_category['cat1'] = cat_1
        data_category['cat2'] = cat_2
    return cat1_count, cat2_count, [data_category['cat1'], data_category['cat2']]

swap_proportion_selected = 0.5
df_global = pd.read_csv(path + 'logging3/Divergence_2_columns_local_{}.csv'.format(data_name))

if not os.path.exists(path_output + data_name):
    os.makedirs(path_output + data_name)
if not os.path.exists(path + path_output + '/processed'):
    os.makedirs(path + path_output + '/processed')
print(len(df_global))
df_global = df_global.dropna(how='any')  # axis='columns'
# print(df_global.columns.tolist())
# df_global = df_global.dropna(subset=['wasserstein_div'], how=-1)
df_global = df_global[df_global.wasserstein_div != -1]
print(len(df_global))
df_global = df_global[df_global.cramers_v_div != -1]
print(len(df_global))
'''Feature = df_global.Feature.values.tolist()
Category = df_global.Category.values.tolist()
swap_proportion = df_global.swap_proportion.values.tolist()
Distortion_hellinger = df_global.Distortion_hellinger.values.tolist()
Distortion_wasserstein = df_global.Distortion_wasserstein.values.tolist()
Distortion_js = df_global.Distortion_js.values.tolist()

hellinger_div = df_global.hellinger_div.values.tolist()
wasserstein_div = df_global.wasserstein_div.values.tolist()
JS_Div = df_global.JS_Div.values.tolist()
Casuality = df_global.Casuality.values.tolist()
Importance = df_global.Importance.values.tolist()
for i in range(len(Feature)):'''

# , 'swap_proportion'

data_file = open(path + path_output + '/processed' + '/Divergence_2_columns_local_{}.csv'.format(data_name),
                 mode='w', newline='',
                 encoding='utf-8')
data_writer = csv.writer(data_file)

Feature = df_global.Feature.values.tolist()

group_data = df_global.groupby(['Feature', 'Feature2', 'swap_proportion', 'Type','Category1'])[
    'distortion_multiple', 'distortion_feature', 'hellinger_div', 'wasserstein_div', 'cramers_v_div', 'total_variation_div', 'JS_Div', 'Casuality', 'Importance', 'effect_distance', 'effect_distance_contrained'].mean().reset_index()

print(set(Feature))

data_file = open(path + path_output + '/processed' + '/Parity_difference_{}.csv'.format(data_name), mode='w',
                  newline='',
                  encoding='utf-8')

data_writer = csv.writer(data_file)
data_writer.writerow(['Features','Swap', 'Swap_ratio', 'Distance_measure','Difference', 'Std', 'Protected', 'NonProtected'])



Feature = group_data.Feature.values.tolist()
Category = group_data.Category1.values.tolist()
Feature2 = group_data.Feature2.values.tolist()
swap_proportion = group_data.swap_proportion.values.tolist()
Type = group_data.Type.values.tolist()
hellinger_div = group_data.hellinger_div.values.tolist()
wasserstein_div = group_data.wasserstein_div.values.tolist()
cramers_v_div = group_data.cramers_v_div.values.tolist()
total_variation_div = group_data.total_variation_div.values.tolist()
JS_Div = group_data.JS_Div.values.tolist()
Casuality = group_data.Casuality.values.tolist()
effect_distance = group_data.effect_distance_contrained.values.tolist()
effect_distance_contrained = group_data.effect_distance.values.tolist()
data_protected_attrib = {}
data_non_protected_attrib = {}

data_protected_attrib_org = {}
data_non_protected_attrib_org = {}
data_dict = {}
for i in range(len(Feature)):
    if Feature[i] in data_dict.keys():
        data_dict[Feature[i]].add(Category[i])
    else:
        data_dict[Feature[i]] = set([Category[i]])
for key, val in data_dict.items():
    list_val = list(val)
    cat1_count, cat2_count, data_category = determin_parity(key, list_val)
    if cat1_count > cat2_count:
        data_protected_attrib[key] = list_val[1]
        data_non_protected_attrib[key] = list_val[0]
        data_protected_attrib_org[key] = data_category[1]
        data_non_protected_attrib_org[key] = data_category[0]
    else:
        data_protected_attrib[key] = list_val[0]
        data_non_protected_attrib[key] = list_val[1]

        data_protected_attrib_org[key] = data_category[0]
        data_non_protected_attrib_org[key] = data_category[1]
# feature_set = set(Feature)
## Direct dictionary
data_direct_hellinger = {}
data_direct_wasserstein = {}
data_direct_cramers_v = {}
data_direct_total_variation = {}
data_direct_JS = {}
data_direct_Casuality = {}
data_direct_effect_distance = {}
data_direct_effect_distance_contrained = {}
## Indirect dictionary
data_indirect_hellinger = {}
data_indirect_wasserstein = {}
data_indirect_cramers_v = {}
data_indirect_total_variation = {}
data_indirect_JS = {}
data_indirect_Casuality = {}
data_indirect_effect_distance = {}
data_indirect_effect_distance_contrained = {}
for i in range(len(Feature)):
    if Feature[i] == Feature2[i]:
        type_ = 'Direct'
        if Feature[i] in data_direct_hellinger.keys():
            if Category[i] in data_direct_hellinger[Feature[i]].keys():
                if swap_proportion[i] in data_direct_hellinger[Feature[i]][Category[i]].keys():
                    if Feature2[i] in data_direct_hellinger[Feature[i]][Category[i]][swap_proportion[i]].keys():
                        
                        data_direct_hellinger[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = hellinger_div[i]
                        data_direct_wasserstein[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = wasserstein_div[i]
                        data_direct_cramers_v[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = cramers_v_div[i]
                        data_direct_total_variation[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                            total_variation_div[i]
                        data_direct_JS[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = JS_Div[i]
                        data_direct_effect_distance[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                            effect_distance[i]
                        data_direct_effect_distance_contrained[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][
                            Type[i]] = effect_distance_contrained[i]
                        data_direct_Casuality[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = Casuality[i]
                    else:
                        data_direct_hellinger[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                        data_direct_hellinger[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = hellinger_div[i]
    
                        data_direct_wasserstein[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                        data_direct_wasserstein[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = wasserstein_div[i]
    
                        data_direct_cramers_v[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                        data_direct_cramers_v[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = cramers_v_div[i]
    
                        data_direct_total_variation[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                        data_direct_total_variation[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                            total_variation_div[i]
    
                        data_direct_JS[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                        data_direct_JS[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = JS_Div[i]
    
                        data_direct_effect_distance[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                        data_direct_effect_distance[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                            effect_distance[i]
    
                        data_direct_effect_distance_contrained[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                        data_direct_effect_distance_contrained[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][
                            Type[i]] = effect_distance_contrained[i]
    
                        data_direct_Casuality[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                        data_direct_Casuality[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = Casuality[i]
                else:
                    data_direct_hellinger[Feature[i]][Category[i]][swap_proportion[i]] = {}
                    data_direct_hellinger[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                    data_direct_hellinger[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = hellinger_div[i]
    
                    data_direct_wasserstein[Feature[i]][Category[i]][swap_proportion[i]] = {}
                    data_direct_wasserstein[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                    data_direct_wasserstein[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = wasserstein_div[i]
    
                    data_direct_cramers_v[Feature[i]][Category[i]][swap_proportion[i]] = {}
                    data_direct_cramers_v[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                    data_direct_cramers_v[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = cramers_v_div[i]
    
                    data_direct_total_variation[Feature[i]][Category[i]][swap_proportion[i]] = {}
                    data_direct_total_variation[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                    data_direct_total_variation[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                        total_variation_div[i]
    
                    data_direct_JS[Feature[i]][Category[i]][swap_proportion[i]] = {}
                    data_direct_JS[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                    data_direct_JS[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = JS_Div[i]
    
                    data_direct_effect_distance[Feature[i]][Category[i]][swap_proportion[i]] = {}
                    data_direct_effect_distance[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                    data_direct_effect_distance[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                        effect_distance[i]
    
                    data_direct_effect_distance_contrained[Feature[i]][Category[i]][swap_proportion[i]] = {}
                    data_direct_effect_distance_contrained[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                    data_direct_effect_distance_contrained[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][
                        Type[i]] = effect_distance_contrained[i]
                    data_direct_Casuality[Feature[i]][Category[i]][swap_proportion[i]] = {}
                    data_direct_Casuality[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                    data_direct_Casuality[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = Casuality[i]
            else:
                data_direct_hellinger[Feature[i]][Category[i]] = {}
                data_direct_hellinger[Feature[i]][Category[i]][swap_proportion[i]] = {}
                data_direct_hellinger[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                data_direct_hellinger[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                hellinger_div[i]

                data_direct_wasserstein[Feature[i]][Category[i]] = {}
                data_direct_wasserstein[Feature[i]][Category[i]][swap_proportion[i]] = {}
                data_direct_wasserstein[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                data_direct_wasserstein[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                wasserstein_div[i]

                data_direct_cramers_v[Feature[i]][Category[i]] = {}
                data_direct_cramers_v[Feature[i]][Category[i]][swap_proportion[i]] = {}
                data_direct_cramers_v[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                data_direct_cramers_v[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                cramers_v_div[i]

                data_direct_total_variation[Feature[i]][Category[i]] = {}
                data_direct_total_variation[Feature[i]][Category[i]][swap_proportion[i]] = {}
                data_direct_total_variation[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                data_direct_total_variation[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                    total_variation_div[i]

                data_direct_JS[Feature[i]][Category[i]] = {}
                data_direct_JS[Feature[i]][Category[i]][swap_proportion[i]] = {}
                data_direct_JS[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                data_direct_JS[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = JS_Div[i]

                data_direct_effect_distance[Feature[i]][Category[i]] = {}
                data_direct_effect_distance[Feature[i]][Category[i]][swap_proportion[i]] = {}
                data_direct_effect_distance[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                data_direct_effect_distance[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                    effect_distance[i]

                data_direct_effect_distance_contrained[Feature[i]][Category[i]] = {}
                data_direct_effect_distance_contrained[Feature[i]][Category[i]][swap_proportion[i]] = {}
                data_direct_effect_distance_contrained[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                data_direct_effect_distance_contrained[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][
                    Type[i]] = effect_distance_contrained[i]

                data_direct_Casuality[Feature[i]][Category[i]] = {}
                data_direct_Casuality[Feature[i]][Category[i]][swap_proportion[i]] = {}
                data_direct_Casuality[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                data_direct_Casuality[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = Casuality[i]
        else:
            data_direct_hellinger[Feature[i]] = {}
            data_direct_hellinger[Feature[i]][Category[i]] = {}
            data_direct_hellinger[Feature[i]][Category[i]][swap_proportion[i]] = {}
            data_direct_hellinger[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
            data_direct_hellinger[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = hellinger_div[i]

            data_direct_wasserstein[Feature[i]] = {}
            data_direct_wasserstein[Feature[i]][Category[i]] = {}
            data_direct_wasserstein[Feature[i]][Category[i]][swap_proportion[i]] = {}
            data_direct_wasserstein[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
            data_direct_wasserstein[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = wasserstein_div[i]

            data_direct_cramers_v[Feature[i]] = {}
            data_direct_cramers_v[Feature[i]][Category[i]] = {}
            data_direct_cramers_v[Feature[i]][Category[i]][swap_proportion[i]] = {}
            data_direct_cramers_v[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
            data_direct_cramers_v[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = cramers_v_div[i]

            data_direct_total_variation[Feature[i]] = {}
            data_direct_total_variation[Feature[i]][Category[i]] = {}
            data_direct_total_variation[Feature[i]][Category[i]][swap_proportion[i]] = {}
            data_direct_total_variation[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
            data_direct_total_variation[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                total_variation_div[i]

            data_direct_JS[Feature[i]] = {}
            data_direct_JS[Feature[i]][Category[i]] = {}
            data_direct_JS[Feature[i]][Category[i]][swap_proportion[i]] = {}
            data_direct_JS[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
            data_direct_JS[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = JS_Div[i]

            data_direct_effect_distance[Feature[i]] = {}
            data_direct_effect_distance[Feature[i]][Category[i]] = {}
            data_direct_effect_distance[Feature[i]][Category[i]][swap_proportion[i]] = {}
            data_direct_effect_distance[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
            data_direct_effect_distance[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                effect_distance[i]

            data_direct_effect_distance_contrained[Feature[i]] = {}
            data_direct_effect_distance_contrained[Feature[i]][Category[i]] = {}
            data_direct_effect_distance_contrained[Feature[i]][Category[i]][swap_proportion[i]] = {}
            data_direct_effect_distance_contrained[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
            data_direct_effect_distance_contrained[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][
                Type[i]] = effect_distance_contrained[i]

            data_direct_Casuality[Feature[i]] = {}
            data_direct_Casuality[Feature[i]][Category[i]] = {}
            data_direct_Casuality[Feature[i]][Category[i]][swap_proportion[i]] = {}
            data_direct_Casuality[Feature[i]][Category[i]][swap_proportion[i]] = {}
            data_direct_Casuality[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
            data_direct_Casuality[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = Casuality[i]

    else:
        type_ = 'Indirect'
        # TODO: Indireect casual effect starts here....
        if Feature[i] in data_indirect_hellinger.keys():
            if Category[i] in data_indirect_hellinger[Feature[i]].keys():
                if swap_proportion[i] in data_indirect_hellinger[Feature[i]][Category[i]].keys():
                    if Feature2[i] in data_indirect_hellinger[Feature[i]][Category[i]][swap_proportion[i]].keys():

                        data_indirect_hellinger[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                        hellinger_div[i]
                        data_indirect_wasserstein[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                        wasserstein_div[i]
                        data_indirect_cramers_v[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                        cramers_v_div[i]
                        data_indirect_total_variation[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                            total_variation_div[i]
                        data_indirect_JS[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = JS_Div[i]
                        data_indirect_effect_distance[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                            effect_distance[i]
                        data_indirect_effect_distance_contrained[Feature[i]][Category[i]][swap_proportion[i]][
                            Feature2[i]][
                            Type[i]] = effect_distance_contrained[i]
                        data_indirect_Casuality[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                        Casuality[i]
                    else:
                        data_indirect_hellinger[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                        data_indirect_hellinger[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                        hellinger_div[i]

                        data_indirect_wasserstein[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                        data_indirect_wasserstein[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                        wasserstein_div[i]

                        data_indirect_cramers_v[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                        data_indirect_cramers_v[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                        cramers_v_div[i]

                        data_indirect_total_variation[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                        data_indirect_total_variation[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                            total_variation_div[i]

                        data_indirect_JS[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                        data_indirect_JS[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = JS_Div[i]

                        data_indirect_effect_distance[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                        data_indirect_effect_distance[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                            effect_distance[i]

                        data_indirect_effect_distance_contrained[Feature[i]][Category[i]][swap_proportion[i]][
                            Feature2[i]] = {}
                        data_indirect_effect_distance_contrained[Feature[i]][Category[i]][swap_proportion[i]][
                            Feature2[i]][
                            Type[i]] = effect_distance_contrained[i]

                        data_indirect_Casuality[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                        data_indirect_Casuality[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                        Casuality[i]
                else:
                    data_indirect_hellinger[Feature[i]][Category[i]][swap_proportion[i]] = {}
                    data_indirect_hellinger[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                    data_indirect_hellinger[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                    hellinger_div[i]

                    data_indirect_wasserstein[Feature[i]][Category[i]][swap_proportion[i]] = {}
                    data_indirect_wasserstein[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                    data_indirect_wasserstein[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                    wasserstein_div[i]

                    data_indirect_cramers_v[Feature[i]][Category[i]][swap_proportion[i]] = {}
                    data_indirect_cramers_v[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                    data_indirect_cramers_v[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                    cramers_v_div[i]

                    data_indirect_total_variation[Feature[i]][Category[i]][swap_proportion[i]] = {}
                    data_indirect_total_variation[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                    data_indirect_total_variation[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                        total_variation_div[i]

                    data_indirect_JS[Feature[i]][Category[i]][swap_proportion[i]] = {}
                    data_indirect_JS[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                    data_indirect_JS[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = JS_Div[i]

                    data_indirect_effect_distance[Feature[i]][Category[i]][swap_proportion[i]] = {}
                    data_indirect_effect_distance[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                    data_indirect_effect_distance[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                        effect_distance[i]

                    data_indirect_effect_distance_contrained[Feature[i]][Category[i]][swap_proportion[i]] = {}
                    data_indirect_effect_distance_contrained[Feature[i]][Category[i]][swap_proportion[i]][
                        Feature2[i]] = {}
                    data_indirect_effect_distance_contrained[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][
                        Type[i]] = effect_distance_contrained[i]
                    data_indirect_Casuality[Feature[i]][Category[i]][swap_proportion[i]] = {}
                    data_indirect_Casuality[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                    data_indirect_Casuality[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                    Casuality[i]
            else:
                data_indirect_hellinger[Feature[i]][Category[i]] = {}
                data_indirect_hellinger[Feature[i]][Category[i]][swap_proportion[i]] = {}
                data_indirect_hellinger[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                data_indirect_hellinger[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                    hellinger_div[i]

                data_indirect_wasserstein[Feature[i]][Category[i]] = {}
                data_indirect_wasserstein[Feature[i]][Category[i]][swap_proportion[i]] = {}
                data_indirect_wasserstein[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                data_indirect_wasserstein[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                    wasserstein_div[i]

                data_indirect_cramers_v[Feature[i]][Category[i]] = {}
                data_indirect_cramers_v[Feature[i]][Category[i]][swap_proportion[i]] = {}
                data_indirect_cramers_v[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                data_indirect_cramers_v[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                    cramers_v_div[i]

                data_indirect_total_variation[Feature[i]][Category[i]] = {}
                data_indirect_total_variation[Feature[i]][Category[i]][swap_proportion[i]] = {}
                data_indirect_total_variation[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                data_indirect_total_variation[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                    total_variation_div[i]

                data_indirect_JS[Feature[i]][Category[i]] = {}
                data_indirect_JS[Feature[i]][Category[i]][swap_proportion[i]] = {}
                data_indirect_JS[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                data_indirect_JS[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = JS_Div[i]

                data_indirect_effect_distance[Feature[i]][Category[i]] = {}
                data_indirect_effect_distance[Feature[i]][Category[i]][swap_proportion[i]] = {}
                data_indirect_effect_distance[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                data_indirect_effect_distance[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                    effect_distance[i]

                data_indirect_effect_distance_contrained[Feature[i]][Category[i]] = {}
                data_indirect_effect_distance_contrained[Feature[i]][Category[i]][swap_proportion[i]] = {}
                data_indirect_effect_distance_contrained[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                data_indirect_effect_distance_contrained[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][
                    Type[i]] = effect_distance_contrained[i]

                data_indirect_Casuality[Feature[i]][Category[i]] = {}
                data_indirect_Casuality[Feature[i]][Category[i]][swap_proportion[i]] = {}
                data_indirect_Casuality[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
                data_indirect_Casuality[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = Casuality[i]
        else:
            data_indirect_hellinger[Feature[i]] = {}
            data_indirect_hellinger[Feature[i]][Category[i]] = {}
            data_indirect_hellinger[Feature[i]][Category[i]][swap_proportion[i]] = {}
            data_indirect_hellinger[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
            data_indirect_hellinger[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = hellinger_div[i]

            data_indirect_wasserstein[Feature[i]] = {}
            data_indirect_wasserstein[Feature[i]][Category[i]] = {}
            data_indirect_wasserstein[Feature[i]][Category[i]][swap_proportion[i]] = {}
            data_indirect_wasserstein[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
            data_indirect_wasserstein[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = wasserstein_div[i]

            data_indirect_cramers_v[Feature[i]] = {}
            data_indirect_cramers_v[Feature[i]][Category[i]] = {}
            data_indirect_cramers_v[Feature[i]][Category[i]][swap_proportion[i]] = {}
            data_indirect_cramers_v[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
            data_indirect_cramers_v[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = cramers_v_div[i]

            data_indirect_total_variation[Feature[i]] = {}
            data_indirect_total_variation[Feature[i]][Category[i]] = {}
            data_indirect_total_variation[Feature[i]][Category[i]][swap_proportion[i]] = {}
            data_indirect_total_variation[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
            data_indirect_total_variation[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                total_variation_div[i]

            data_indirect_JS[Feature[i]] = {}
            data_indirect_JS[Feature[i]][Category[i]] = {}
            data_indirect_JS[Feature[i]][Category[i]][swap_proportion[i]] = {}
            data_indirect_JS[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
            data_indirect_JS[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = JS_Div[i]

            data_indirect_effect_distance[Feature[i]] = {}
            data_indirect_effect_distance[Feature[i]][Category[i]] = {}
            data_indirect_effect_distance[Feature[i]][Category[i]][swap_proportion[i]] = {}
            data_indirect_effect_distance[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
            data_indirect_effect_distance[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                effect_distance[i]

            data_indirect_effect_distance_contrained[Feature[i]] = {}
            data_indirect_effect_distance_contrained[Feature[i]][Category[i]] = {}
            data_indirect_effect_distance_contrained[Feature[i]][Category[i]][swap_proportion[i]] = {}
            data_indirect_effect_distance_contrained[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
            data_indirect_effect_distance_contrained[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][
                Type[i]] = effect_distance_contrained[i]

            data_indirect_Casuality[Feature[i]] = {}
            data_indirect_Casuality[Feature[i]][Category[i]] = {}
            data_indirect_Casuality[Feature[i]][Category[i]][swap_proportion[i]] = {}
            data_indirect_Casuality[Feature[i]][Category[i]][swap_proportion[i]] = {}
            data_indirect_Casuality[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]] = {}
            data_indirect_Casuality[Feature[i]][Category[i]][swap_proportion[i]][Feature2[i]][Type[i]] = Casuality[i]
row = ['Swap \n Percentage', 'Measurement', 'Feature']
# feature_set = set(['race', 'sex', 'age', 'hours-per-week', 'capital-gain', 'capital-loss'])
# feature_list = ['race', 'sex', 'age', 'hours-per-week', 'capital-gain', 'capital-loss'] #list(feature_set)
feature_list = ['sex','age',  'thalach', 'ca', 'thal', 'exang', 'cp', 'trestbps', 'restecg', 'fbs', 'oldpeak', 'chol'] #list(feature_set)
# feature_list = ['race', 'sex', 'age', 'hours-per-week', 'capital-gain', 'capital-loss'] #list(feature_set)
# feature_list = ['sex','age','health','Pstatus','nursery','Medu', 'Fjob', 'schoolsup', 'absences',  'activities', 'higher', 'traveltime',  'paid', 'guardian',  'Walc', 'freetime', 'famsup',  'romantic', 'studytime', 'goout', 'reason',  'famrel', 'internet']
# feature_list = [ 'Sex', 'Age','Job', 'Saving', 'Checking', 'Credit','Housing', 'Purpose']
feature_list = ['race', 'sex', 'age', 'c_charge_degree', 'priors_count']
#feature_list = ['age', 'education', 'job', 'loan', 'balance', 'housing', 'duration', 'campaign', 'default']

feature_set = set(feature_list)

data_probability_mapping_CDI = {}
data_probability_mapping_NDI = {}
data_probability_mapping_NII = {}
data_probability_mapping_NDI_NII = {}
for feature_ in feature_set:
    # for feature_, val in data_direct_hellinger.items():
    #print(feature_, data_direct_hellinger)
    #if feature_ in data_direct_hellinger.keys():
    #print(data_direct_hellinger.keys())
    for category in data_direct_hellinger[feature_].keys():
        val = data_direct_hellinger[feature_][category]
        # print(feature_, val)
        for swap_, val2_ in val.items():
            type_ = 'CDI'
            swap_prop = int(swap_ * 100)
            direct_effect_hellinger = np.mean(val2_[feature_][type_])
            direct_effect_wasserstein = np.mean(data_direct_wasserstein[feature_][category][swap_][feature_][type_])
            direct_effect_cramers_v = np.mean(data_direct_cramers_v[feature_][category][swap_][feature_][type_])
            direct_effect_total_variation = np.mean(data_direct_total_variation[feature_][category][swap_][feature_][type_])
            direct_effect_JS = np.mean(data_direct_JS[feature_][category][swap_][feature_][type_])
            direct_effect_Casuality = np.mean(data_direct_Casuality[feature_][category][swap_][feature_][type_])
            direct_effect_distance = np.mean(data_direct_effect_distance[feature_][category][swap_][feature_][type_])
            direct_effect_distance_contrained = np.mean(
                data_direct_effect_distance_contrained[feature_][category][swap_][feature_][type_])
            if not feature_ in data_probability_mapping_CDI.keys():
                data_probability_mapping_CDI[feature_] = {}
            if not swap_prop in data_probability_mapping_CDI[feature_].keys():
                data_probability_mapping_CDI[feature_][swap_prop] = {}
            if not category in data_probability_mapping_CDI[feature_][swap_prop].keys():
                data_probability_mapping_CDI[feature_][swap_prop][category] = {}
            data_probability_mapping_CDI[feature_][swap_prop][category]['hellinger'] = direct_effect_hellinger
            data_probability_mapping_CDI[feature_][swap_prop][category]['wasserstein'] = direct_effect_wasserstein
            data_probability_mapping_CDI[feature_][swap_prop][category]['cramers_v'] = direct_effect_cramers_v
            data_probability_mapping_CDI[feature_][swap_prop][category]['total_variation'] = direct_effect_total_variation
            data_probability_mapping_CDI[feature_][swap_prop][category]['JS'] = direct_effect_JS
            data_probability_mapping_CDI[feature_][swap_prop][category]['Casuality'] = direct_effect_Casuality
            data_probability_mapping_CDI[feature_][swap_prop][category]['effect_distance'] = direct_effect_distance
            data_probability_mapping_CDI[feature_][swap_prop][category]['distance_contrained'] = direct_effect_distance_contrained
            # print(swap_)
            for feature_2 in feature_set:
                # for feature_2, val2 in data_indirect_hellinger[feature_][swap_].items():
                if feature_ in data_indirect_hellinger.keys():
                    #print(data_indirect_hellinger[feature_])
                    if category in data_indirect_hellinger[feature_].keys():
                        if feature_2 != feature_ and feature_2 in data_indirect_hellinger[feature_][category][
                            swap_].keys():  # feature_list.index(feature_) <= feature_list.index(feature_2):
                            # print(feature_, feature_2, data_indirect_hellinger[feature_])
                            val2 = data_indirect_hellinger[feature_][category][swap_][feature_2]
                            # todo: direct impact results 1
                            type_NDI = 'NDI'
                            type_NII = 'NII'

                            # print('val2: ', val2)
                            # for type_, val2 in val3.items():

                            # todo: NDI results 1
                            # print(data_indirect_hellinger)
                            NDI_effect_hellinger = np.mean(data_indirect_hellinger[feature_][category][swap_][feature_2][type_NDI])
                            NDI_effect_wasserstein = np.mean(data_indirect_wasserstein[feature_][category][swap_][feature_2][type_NDI])
                            NDI_effect_cramers_v = np.mean(data_indirect_cramers_v[feature_][category][swap_][feature_2][type_NDI])
                            NDI_effect_total_variation = np.mean(
                                data_indirect_total_variation[feature_][category][swap_][feature_2][type_NDI])
                            # print(data_indirect_JS[feature_][swap_], data_indirect_JS[feature_][swap_][keys_[1]])
                            NDI_effect_JS = np.mean(data_indirect_JS[feature_][category][swap_][feature_2][type_NDI])
                            # print(data_indirect_Casuality[feature_][swap_], data_indirect_Casuality[feature_][swap_][keys_[1]])
                            NDI_effect_Casuality = np.mean(data_indirect_Casuality[feature_][category][swap_][feature_2][type_NDI])
                            NDI_effect_distance = np.mean(data_indirect_effect_distance[feature_][category][swap_][feature_2][type_NDI])
                            NDI_effect_distance_contrained = np.mean(
                                data_indirect_effect_distance_contrained[feature_][category][swap_][feature_2][type_NDI])

                            # todo: NII results 1
                            #print(data_indirect_hellinger[feature_][category][swap_][feature_2])
                            if type_NII in data_indirect_hellinger[feature_][category][swap_][feature_2].keys():
                                NII_effect_hellinger = np.mean(data_indirect_hellinger[feature_][category][swap_][feature_2][type_NII])
                                NII_effect_wasserstein = np.mean(data_indirect_wasserstein[feature_][category][swap_][feature_2][type_NII])
                                NII_effect_cramers_v = np.mean(data_indirect_cramers_v[feature_][category][swap_][feature_2][type_NII])
                                NII_effect_total_variation = np.mean(
                                    data_indirect_total_variation[feature_][category][swap_][feature_2][type_NII])
                                # print(data_indirect_JS[feature_][swap_], data_indirect_JS[feature_][swap_][keys_[1]])
                                NII_effect_JS = np.mean(data_indirect_JS[feature_][category][swap_][feature_2][type_NII])
                                # print(data_indirect_Casuality[feature_][swap_], data_indirect_Casuality[feature_][swap_][keys_[1]])
                                NII_effect_Casuality = np.mean(data_indirect_Casuality[feature_][category][swap_][feature_2][type_NII])
                                NII_effect_distance = np.mean(data_indirect_effect_distance[feature_][category][swap_][feature_2][type_NII])
                                NII_effect_distance_contrained = np.mean(
                                    data_indirect_effect_distance_contrained[feature_][category][swap_][feature_2][type_NII])
                            else:
                                NII_effect_hellinger, NII_effect_wasserstein, NII_effect_cramers_v, NII_effect_total_variation, NII_effect_JS, NII_effect_Casuality, NII_effect_distance, NII_effect_distance_contrained = 0, 0,0,0,0,0,0,0
                            if not feature_ + '/'+ category + '/' + feature_2 in data_probability_mapping_NDI.keys():
                                data_probability_mapping_NDI[feature_ +'/' + category + '/' + feature_2] = {}
                            if not swap_prop in data_probability_mapping_NDI[feature_ + '/' + category + '/' + feature_2].keys():
                                data_probability_mapping_NDI[feature_ +'/'+ category + '/' + feature_2][swap_prop] = {}
                            data_probability_mapping_NDI[feature_ + '/'+ category + '/' + feature_2][swap_prop][
                                'hellinger'] = NDI_effect_hellinger
                            data_probability_mapping_NDI[feature_ + '/'+ category + '/' + feature_2][swap_prop][
                                'wasserstein'] = NDI_effect_wasserstein
                            data_probability_mapping_NDI[feature_ + '/'+ category + '/' + feature_2][swap_prop][
                                'cramers_v'] = NDI_effect_cramers_v
                            data_probability_mapping_NDI[feature_ + '/'+ category + '/' + feature_2][swap_prop][
                                'total_variation'] = NDI_effect_total_variation
                            data_probability_mapping_NDI[feature_ + '/'+ category + '/' + feature_2][swap_prop]['JS'] = NDI_effect_JS
                            data_probability_mapping_NDI[feature_ + '/'+ category + '/' + feature_2][swap_prop][
                                'Casuality'] = NDI_effect_Casuality
                            data_probability_mapping_NDI[feature_ + '/'+ category + '/' + feature_2][swap_prop][
                                'effect_distance'] = NDI_effect_distance
                            data_probability_mapping_NDI[feature_ + '/'+ category + '/' + feature_2][swap_prop][
                                'distance_contrained'] = NDI_effect_distance_contrained

                            if not feature_ + '/'+ category + '/' + feature_2 in data_probability_mapping_NII.keys():
                                data_probability_mapping_NII[feature_ + '/'+ category + '/' + feature_2] = {}
                            if not swap_prop in data_probability_mapping_NII[feature_ + '/'+ category + '/' + feature_2].keys():
                                data_probability_mapping_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop] = {}

                            # data_probability_mapping_NII[feature_+'/'+feature_2] = {}
                            # data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop] = {}
                            data_probability_mapping_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop][
                                'hellinger'] = NII_effect_hellinger
                            data_probability_mapping_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop][
                                'wasserstein'] = NII_effect_wasserstein
                            data_probability_mapping_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop][
                                'cramers_v'] = NII_effect_cramers_v
                            data_probability_mapping_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop][
                                'total_variation'] = NII_effect_total_variation
                            data_probability_mapping_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop]['JS'] = NII_effect_JS
                            data_probability_mapping_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop][
                                'Casuality'] = NII_effect_Casuality
                            data_probability_mapping_NII[feature_ + '/' + category + '/'+ feature_2][swap_prop][
                                'effect_distance'] = NII_effect_distance
                            data_probability_mapping_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop][
                                'distance_contrained'] = NII_effect_distance_contrained

                            ##todo: Combining the two
                            '''if not feature_ + '/' + feature_2 in data_probability_mapping_NDI_NII.keys():
                                data_probability_mapping_NDI_NII[feature_ + '/' + feature_2] = {}
                            if not swap_prop in data_probability_mapping_NDI_NII[feature_ + '/' + feature_2].keys():
                                data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop] = {}
                            data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop]['hellinger'] = NDI_effect_hellinger + NII_effect_hellinger
                            data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop][
                                'wasserstein'] = NDI_effect_wasserstein+NII_effect_wasserstein
                            data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop]['cramers_v'] = NDI_effect_cramers_v+NII_effect_cramers_v
                            data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop][
                                'total_variation'] = NDI_effect_total_variation+NII_effect_total_variation
                            data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop]['JS'] = NDI_effect_JS+NII_effect_JS
                            data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop]['Casuality'] = NDI_effect_Casuality+NII_effect_Casuality
                            data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop][
                                'effect_distance'] = NDI_effect_distance+NII_effect_distance
                            data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop][
                                'distance_contrained'] = NDI_effect_distance_contrained+NII_effect_distance_contrained'''
                            if not feature_ + '/'+ category + '/' + feature_2 in data_probability_mapping_NDI_NII.keys():
                                data_probability_mapping_NDI_NII[feature_ + '/'+ category + '/' + feature_2] = {}
                            if not swap_prop in data_probability_mapping_NDI_NII[feature_ + '/'+ category + '/' + feature_2].keys():
                                data_probability_mapping_NDI_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop] = {}
                            data_probability_mapping_NDI_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop][
                                'hellinger'] = NDI_effect_hellinger  # + NII_effect_hellinger
                            data_probability_mapping_NDI_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop][
                                'wasserstein'] = NDI_effect_wasserstein  # +NII_effect_wasserstein
                            data_probability_mapping_NDI_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop][
                                'cramers_v'] = NDI_effect_cramers_v  # +NII_effect_cramers_v
                            data_probability_mapping_NDI_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop][
                                'total_variation'] = NDI_effect_total_variation + NII_effect_total_variation
                            data_probability_mapping_NDI_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop][
                                'JS'] = NDI_effect_JS  # +NII_effect_JS
                            data_probability_mapping_NDI_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop][
                                'Casuality'] = NDI_effect_Casuality  # +NII_effect_Casuality
                            data_probability_mapping_NDI_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop][
                                'effect_distance'] = NDI_effect_distance  # +NII_effect_distance
                            data_probability_mapping_NDI_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop][
                                'distance_contrained'] = NDI_effect_distance_contrained  # +NII_effect_distance_contrained

swap_selected = [10, 30, 50, 70]
for feature_ in feature_set:

    val = data_probability_mapping_CDI.get(feature_)
    # print(val.keys())
    #if val != None:
    print(feature_, val)
    for swap_prop, val2_ in val.items():
        if swap_prop in swap_selected and feature_list.index(feature_) < len(feature_list)-2:

            value_hellinger_NDI = {}
            value_wasserstein_NDI = {}
            value_cramers_v_NDI = {}
            value_total_variation_NDI = {}
            value_js_NDI = {}
            value_Casuality_NDI = {}
            value_effect_distance_NDI = {}
            value_effect_distance_constrained_NDI = {}

            value_hellinger_NII = {}
            value_wasserstein_NII = {}
            value_cramers_v_NII = {}
            value_total_variation_NII = {}
            value_js_NII = {}
            value_Casuality_NII = {}
            value_effect_distance_NII = {}
            value_effect_distance_constrained_NII = {}

            value_hellinger_NDI_NII = {}
            value_wasserstein_NDI_NII = {}
            value_cramers_v_NDI_NII = {}
            value_total_variation_NDI_NII = {}
            value_js_NDI_NII = {}

            value_Casuality_NDI_NII = {}
            value_effect_distance_NDI_NII = {}
            value_effect_distance_constrained_NDI_NII = {}

            # print(val, data_probability_mapping_CDI)

            for category in data_probability_mapping_CDI[feature_][swap_prop].keys():

                for feature_2 in feature_set:
                    if feature_ + '/' + category + '/' + feature_2 in data_probability_mapping_NDI.keys():
                        ## todo: Natural indirect impact
                        val_h = 0
                        val_w = 0
                        val_c = 0
                        val_t = 0
                        val_s = 0

                        val_cas = 0
                        val_eff = 0
                        val_dCons = 0

                        if feature_list.index(feature_) < feature_list.index(feature_2):
                            if category in value_hellinger_NDI_NII.keys():
                                #value_hellinger_NDI_NII[category]
                                value_hellinger_NII[category].append(
                                    data_probability_mapping_NII[feature_ + '/'+ category + '/'  + feature_2][swap_prop]['hellinger'])
                                val_h = data_probability_mapping_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop]['hellinger']
                                # if data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop]['wasserstein'] < \
                                #         data_probability_mapping_NII[feature_2 + '/' + feature_][swap_prop][
                                #             'wasserstein'] and feature_list.index(feature_) < feature_list.index(feature_2):
                                value_wasserstein_NII[category].append(
                                    data_probability_mapping_NII[feature_ + '/' + category + '/'+ feature_2][swap_prop]['wasserstein'])
                                val_w = data_probability_mapping_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop]['wasserstein']
                                # if data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop]['cramers_v'] < \
                                #         data_probability_mapping_NII[feature_2 + '/' + feature_][swap_prop][
                                #             'cramers_v'] and feature_list.index(feature_) < feature_list.index(feature_2):
                                value_cramers_v_NII[category].append(
                                    data_probability_mapping_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop]['cramers_v'])
                                val_c = data_probability_mapping_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop]['cramers_v']
                                # if data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop]['total_variation'] < \
                                #         data_probability_mapping_NII[feature_2 + '/' + feature_][swap_prop][
                                #             'total_variation'] and feature_list.index(feature_) < feature_list.index(feature_2):
                                value_total_variation_NII[category].append(
                                    data_probability_mapping_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop]['total_variation'])
                                val_t = data_probability_mapping_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop]['total_variation']
                                # if data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop]['JS'] < \
                                #         data_probability_mapping_NII[feature_2 + '/' + feature_][swap_prop][
                                #             'JS'] and feature_list.index(feature_) < feature_list.index(feature_2):
                                value_js_NII[category].append(data_probability_mapping_NII[feature_ + '/' + category + '/'+ feature_2][swap_prop]['JS'])
                                val_s = data_probability_mapping_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop]['JS']

                                ## Added temporary
                                # if data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop]['Casuality'] < \
                                #         data_probability_mapping_NII[feature_2 + '/' + feature_][swap_prop][
                                #             'Casuality'] and feature_list.index(feature_) < feature_list.index(feature_2):
                                value_Casuality_NII[category].append(
                                    data_probability_mapping_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop]['Casuality'])
                                val_cas = data_probability_mapping_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop]['Casuality']

                                # if data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop]['effect_distance'] < \
                                #         data_probability_mapping_NII[feature_2 + '/' + feature_][swap_prop][
                                #             'effect_distance'] and feature_list.index(feature_) < feature_list.index(feature_2):
                                value_effect_distance_NII.append(
                                    data_probability_mapping_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop]['effect_distance'])
                                val_eff = data_probability_mapping_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop]['effect_distance']
                                # if data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop]['distance_contrained'] < \
                                #         data_probability_mapping_NII[feature_2 + '/' + feature_][swap_prop][
                                #             'distance_contrained'] and feature_list.index(feature_) < feature_list.index(feature_2):
                                value_effect_distance_constrained_NII[category].append(
                                    data_probability_mapping_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop]['JS'])
                                val_dCons = data_probability_mapping_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop][
                                    'distance_contrained']

                                ## todo: Combined Natural direct and Indirect impact
                                # if data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop]['hellinger'] < \
                                #         data_probability_mapping_NDI_NII[feature_2 + '/' + feature_][swap_prop][
                                #             'hellinger'] and feature_list.index(feature_) < feature_list.index(feature_2):
                                value_hellinger_NDI_NII[category].append(
                                    data_probability_mapping_NDI_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop][
                                        'hellinger'] + val_h)
                                # if data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop]['wasserstein'] < \
                                #         data_probability_mapping_NDI_NII[feature_2 + '/' + feature_][swap_prop][
                                #             'wasserstein'] and feature_list.index(feature_) < feature_list.index(feature_2):
                                value_wasserstein_NDI_NII[category].append(
                                    data_probability_mapping_NDI_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop][
                                        'wasserstein'] + val_w)
                                # if data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop]['cramers_v'] < \
                                #         data_probability_mapping_NDI_NII[feature_2 + '/' + feature_][swap_prop][
                                #             'cramers_v'] and feature_list.index(feature_) < feature_list.index(feature_2):
                                value_cramers_v_NDI_NII[category].append(
                                    data_probability_mapping_NDI_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop][
                                        'cramers_v'] + val_c)
                                # if data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop]['total_variation'] < \
                                #         data_probability_mapping_NDI_NII[feature_2 + '/' + feature_][swap_prop][
                                #             'total_variation'] and feature_list.index(feature_) < feature_list.index(feature_2):
                                value_total_variation_NDI_NII[category].append(
                                    data_probability_mapping_NDI_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop][
                                        'total_variation'] + val_t)
                                # if data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop]['JS'] < \
                                #         data_probability_mapping_NDI_NII[feature_2 + '/' + feature_][swap_prop][
                                #             'JS'] and feature_list.index(feature_) < feature_list.index(feature_2):
                                value_js_NDI_NII[category].append(
                                    data_probability_mapping_NDI_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop]['JS'] + val_s)

                                # temporary added
                                # if data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop]['Casuality'] < \
                                #         data_probability_mapping_NDI_NII[feature_2 + '/' + feature_][swap_prop][
                                #             'Casuality'] and feature_list.index(feature_) < feature_list.index(feature_2):
                                value_Casuality_NDI_NII[category].append(
                                    data_probability_mapping_NDI_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop][
                                        'Casuality'] + val_cas)
                                # if data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop]['effect_distance'] < \
                                #         data_probability_mapping_NDI_NII[feature_2 + '/' + feature_][swap_prop][
                                #             'effect_distance'] and feature_list.index(feature_) < feature_list.index(feature_2):
                                value_effect_distance_NDI_NII[category].append(
                                    data_probability_mapping_NDI_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop][
                                        'effect_distance'] + val_eff)
                                # if data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop]['distance_contrained'] < \
                                #       data_probability_mapping_NDI_NII[feature_2 + '/' + feature_][swap_prop][
                                #            'distance_contrained'] and feature_list.index(feature_) < feature_list.index(feature_2):
                                value_effect_distance_constrained_NDI_NII[category].append(
                                    data_probability_mapping_NDI_NII[feature_ + '/'+ category + '/' + feature_2][swap_prop][
                                        'distance_contrained'] + val_dCons)
                            else:

                                value_hellinger_NII[category] = [
                                    data_probability_mapping_NII[feature_ + '/' + category + '/' + feature_2][
                                        swap_prop]['hellinger']]
                                val_h = \
                                data_probability_mapping_NII[feature_ + '/' + category + '/' + feature_2][swap_prop][
                                    'hellinger']
                                # if data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop]['wasserstein'] < \
                                #         data_probability_mapping_NII[feature_2 + '/' + feature_][swap_prop][
                                #             'wasserstein'] and feature_list.index(feature_) < feature_list.index(feature_2):
                                value_wasserstein_NII[category] = [
                                    data_probability_mapping_NII[feature_ + '/' + category + '/' + feature_2][
                                        swap_prop]['wasserstein']]
                                val_w = \
                                data_probability_mapping_NII[feature_ + '/' + category + '/' + feature_2][swap_prop][
                                    'wasserstein']
                                # if data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop]['cramers_v'] < \
                                #         data_probability_mapping_NII[feature_2 + '/' + feature_][swap_prop][
                                #             'cramers_v'] and feature_list.index(feature_) < feature_list.index(feature_2):
                                value_cramers_v_NII[category] = [
                                    data_probability_mapping_NII[feature_ + '/' + category + '/' + feature_2][
                                        swap_prop]['cramers_v']]
                                val_c = \
                                data_probability_mapping_NII[feature_ + '/' + category + '/' + feature_2][swap_prop][
                                    'cramers_v']
                                # if data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop]['total_variation'] < \
                                #         data_probability_mapping_NII[feature_2 + '/' + feature_][swap_prop][
                                #             'total_variation'] and feature_list.index(feature_) < feature_list.index(feature_2):
                                value_total_variation_NII[category] = [
                                    data_probability_mapping_NII[feature_ + '/' + category + '/' + feature_2][
                                        swap_prop]['total_variation']]
                                val_t = \
                                data_probability_mapping_NII[feature_ + '/' + category + '/' + feature_2][swap_prop][
                                    'total_variation']
                                # if data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop]['JS'] < \
                                #         data_probability_mapping_NII[feature_2 + '/' + feature_][swap_prop][
                                #             'JS'] and feature_list.index(feature_) < feature_list.index(feature_2):
                                value_js_NII[category] = [
                                    data_probability_mapping_NII[feature_ + '/' + category + '/' + feature_2][
                                        swap_prop]['JS']]
                                val_s = \
                                data_probability_mapping_NII[feature_ + '/' + category + '/' + feature_2][swap_prop][
                                    'JS']

                                ## Added temporary
                                # if data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop]['Casuality'] < \
                                #         data_probability_mapping_NII[feature_2 + '/' + feature_][swap_prop][
                                #             'Casuality'] and feature_list.index(feature_) < feature_list.index(feature_2):
                                value_Casuality_NII[category] = [
                                    data_probability_mapping_NII[feature_ + '/' + category + '/' + feature_2][
                                        swap_prop]['Casuality']]
                                val_cas = \
                                data_probability_mapping_NII[feature_ + '/' + category + '/' + feature_2][swap_prop][
                                    'Casuality']

                                # if data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop]['effect_distance'] < \
                                #         data_probability_mapping_NII[feature_2 + '/' + feature_][swap_prop][
                                #             'effect_distance'] and feature_list.index(feature_) < feature_list.index(feature_2):
                                value_effect_distance_NII = [
                                    data_probability_mapping_NII[feature_ + '/' + category + '/' + feature_2][
                                        swap_prop]['effect_distance']]
                                val_eff = \
                                data_probability_mapping_NII[feature_ + '/' + category + '/' + feature_2][swap_prop][
                                    'effect_distance']
                                # if data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop]['distance_contrained'] < \
                                #         data_probability_mapping_NII[feature_2 + '/' + feature_][swap_prop][
                                #             'distance_contrained'] and feature_list.index(feature_) < feature_list.index(feature_2):
                                value_effect_distance_constrained_NII[category] = [
                                    data_probability_mapping_NII[feature_ + '/' + category + '/' + feature_2][
                                        swap_prop]['JS']]
                                val_dCons = \
                                data_probability_mapping_NII[feature_ + '/' + category + '/' + feature_2][swap_prop][
                                    'distance_contrained']

                                ## todo: Combined Natural direct and Indirect impact
                                # if data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop]['hellinger'] < \
                                #         data_probability_mapping_NDI_NII[feature_2 + '/' + feature_][swap_prop][
                                #             'hellinger'] and feature_list.index(feature_) < feature_list.index(feature_2):
                                value_hellinger_NDI_NII[category] = [
                                    data_probability_mapping_NDI_NII[feature_ + '/' + category + '/' + feature_2][
                                        swap_prop][
                                        'hellinger'] + val_h]
                                # if data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop]['wasserstein'] < \
                                #         data_probability_mapping_NDI_NII[feature_2 + '/' + feature_][swap_prop][
                                #             'wasserstein'] and feature_list.index(feature_) < feature_list.index(feature_2):
                                value_wasserstein_NDI_NII[category] = [
                                    data_probability_mapping_NDI_NII[feature_ + '/' + category + '/' + feature_2][
                                        swap_prop][
                                        'wasserstein'] + val_w]
                                # if data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop]['cramers_v'] < \
                                #         data_probability_mapping_NDI_NII[feature_2 + '/' + feature_][swap_prop][
                                #             'cramers_v'] and feature_list.index(feature_) < feature_list.index(feature_2):
                                value_cramers_v_NDI_NII[category] = [
                                    data_probability_mapping_NDI_NII[feature_ + '/' + category + '/' + feature_2][
                                        swap_prop][
                                        'cramers_v'] + val_c]
                                # if data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop]['total_variation'] < \
                                #         data_probability_mapping_NDI_NII[feature_2 + '/' + feature_][swap_prop][
                                #             'total_variation'] and feature_list.index(feature_) < feature_list.index(feature_2):
                                value_total_variation_NDI_NII[category] = [
                                    data_probability_mapping_NDI_NII[feature_ + '/' + category + '/' + feature_2][
                                        swap_prop][
                                        'total_variation'] + val_t]
                                # if data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop]['JS'] < \
                                #         data_probability_mapping_NDI_NII[feature_2 + '/' + feature_][swap_prop][
                                #             'JS'] and feature_list.index(feature_) < feature_list.index(feature_2):
                                value_js_NDI_NII[category] = [
                                    data_probability_mapping_NDI_NII[feature_ + '/' + category + '/' + feature_2][
                                        swap_prop]['JS'] + val_s]

                                # temporary added
                                # if data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop]['Casuality'] < \
                                #         data_probability_mapping_NDI_NII[feature_2 + '/' + feature_][swap_prop][
                                #             'Casuality'] and feature_list.index(feature_) < feature_list.index(feature_2):
                                value_Casuality_NDI_NII[category] = [
                                    data_probability_mapping_NDI_NII[feature_ + '/' + category + '/' + feature_2][
                                        swap_prop][
                                        'Casuality'] + val_cas]
                                # if data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop]['effect_distance'] < \
                                #         data_probability_mapping_NDI_NII[feature_2 + '/' + feature_][swap_prop][
                                #             'effect_distance'] and feature_list.index(feature_) < feature_list.index(feature_2):
                                value_effect_distance_NDI_NII[category] = [
                                    data_probability_mapping_NDI_NII[feature_ + '/' + category + '/' + feature_2][
                                        swap_prop][
                                        'effect_distance'] + val_eff]
                                # if data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop]['distance_contrained'] < \
                                #       data_probability_mapping_NDI_NII[feature_2 + '/' + feature_][swap_prop][
                                #            'distance_contrained'] and feature_list.index(feature_) < feature_list.index(feature_2):
                                value_effect_distance_constrained_NDI_NII[category] = [
                                    data_probability_mapping_NDI_NII[feature_ + '/' + category + '/' + feature_2][
                                        swap_prop][
                                        'distance_contrained'] + val_dCons]




            '''hellinger_NDI_mean = 0
            wasserstein_NDI_mean = 0
            cramers_v_NDI_mean = 0
            total_variation_NDI_mean = 0
            JS_NDI_mean = 0

            Casuality_NDI_mean = 0
            effect_distance_NDI_mean = 0
            effect_distance_constained_NDI_mean = 0
            if len(value_hellinger_NDI) > 0:
                hellinger_NDI_mean = np.sum(value_hellinger_NDI)
            if len(value_wasserstein_NDI) > 0:
                wasserstein_NDI_mean = np.sum(value_wasserstein_NDI)
            if len(value_cramers_v_NDI) > 0:
                cramers_v_NDI_mean = np.sum(value_cramers_v_NDI)
            if len(value_total_variation_NDI) > 0:
                total_variation_NDI_mean = np.sum(value_total_variation_NDI)
            if len(value_js_NDI) > 0:
                JS_NDI_mean = np.sum(value_js_NDI)

            if len(value_Casuality_NDI) > 0:
                Casuality_NDI_mean = np.sum(value_Casuality_NDI)
            if len(value_effect_distance_NDI) > 0:
                effect_distance_NDI_mean = np.sum(value_effect_distance_NDI)
            if len(value_effect_distance_constrained_NDI) > 0:
                effect_distance_constained_NDI_mean = np.sum(value_effect_distance_constrained_NDI)
            # todo: natual indirect
            hellinger_NII_mean = 0
            wasserstein_NII_mean = 0
            cramers_v_NII_mean = 0
            total_variation_NII_mean = 0
            JS_NII_mean = 0

            Casuality_NII_mean = 0
            effect_distance_NII_mean = 0
            effect_distance_constained_NII_mean = 0
            if len(value_hellinger_NII) > 0:
                hellinger_NII_mean = np.sum(value_hellinger_NII)
            if len(value_wasserstein_NII) > 0:
                wasserstein_NII_mean = np.sum(value_wasserstein_NII)
            if len(value_cramers_v_NII) > 0:
                cramers_v_NII_mean = np.sum(value_cramers_v_NII)
            if len(value_total_variation_NII) > 0:
                total_variation_NII_mean = np.sum(value_total_variation_NII)
            if len(value_js_NII) > 0:
                JS_NII_mean = np.sum(value_js_NII)

            if len(value_Casuality_NII) > 0:
                Casuality_NII_mean = np.sum(value_Casuality_NII)
            if len(value_effect_distance_NII) > 0:
                effect_distance_NII_mean = np.sum(value_effect_distance_NII)
            if len(value_effect_distance_constrained_NII) > 0:
                effect_distance_constained_NII_mean = np.sum(value_effect_distance_constrained_NII)'''
            # todo: combined natural direct and indirect impact
            hellinger_NDI_NII_mean_protected = 0
            wasserstein_NDI_NII_mean_protected = 0
            cramers_v_NDI_NII_mean_protected = 0
            total_variation_NDI_NII_mean_protected = 0
            JS_NDI_NII_mean_protected = 0

            hellinger_NDI_NII_mean_non_protected = 0
            wasserstein_NDI_NII_mean_non_protected = 0
            cramers_v_NDI_NII_mean_non_protected = 0
            total_variation_NDI_NII_mean_non_protected = 0
            JS_NDI_NII_mean_non_protected = 0
            val_protected, val_non_protected = None, None
            print(value_hellinger_NDI_NII)
            if data_protected_attrib.get(feature_) in value_hellinger_NDI_NII.keys():
                #val_protected = val2[data_protected_attrib.get(key)]
                if len(value_hellinger_NDI_NII[data_protected_attrib.get(feature_)]) > 0:
                    hellinger_NDI_NII_mean_protected = np.sum(value_hellinger_NDI_NII[data_protected_attrib.get(feature_)])
                if len(value_wasserstein_NDI_NII[data_protected_attrib.get(feature_)]) > 0:
                    wasserstein_NDI_NII_mean_protected = np.sum(value_wasserstein_NDI_NII[data_protected_attrib.get(feature_)])
                if len(value_cramers_v_NDI_NII[data_protected_attrib.get(feature_)]) > 0:
                    cramers_v_NDI_NII_mean_protected = np.sum(value_cramers_v_NDI_NII[data_protected_attrib.get(feature_)])
                if len(value_total_variation_NDI_NII[data_protected_attrib.get(feature_)]) > 0:
                    total_variation_NDI_NII_mean_protected = np.sum(value_total_variation_NDI_NII[data_protected_attrib.get(feature_)])
                if len(value_js_NDI_NII[data_protected_attrib.get(feature_)]) > 0:
                    JS_NDI_NII_mean_protected = np.sum(value_js_NDI_NII[data_protected_attrib.get(feature_)])

            if data_non_protected_attrib.get(feature_) in value_hellinger_NDI_NII.keys():
                #val_non_protected = val2[data_non_protected_attrib.get(key)]
                if len(value_hellinger_NDI_NII[data_non_protected_attrib.get(feature_)]) > 0:
                    hellinger_NDI_NII_mean_non_protected = np.sum(
                        value_hellinger_NDI_NII[data_non_protected_attrib.get(feature_)])
                if len(value_wasserstein_NDI_NII[data_non_protected_attrib.get(feature_)]) > 0:
                    wasserstein_NDI_NII_mean_non_protected = np.sum(
                        value_wasserstein_NDI_NII[data_non_protected_attrib.get(feature_)])
                if len(value_cramers_v_NDI_NII[data_non_protected_attrib.get(feature_)]) > 0:
                    cramers_v_NDI_NII_mean_non_protected = np.sum(
                        value_cramers_v_NDI_NII[data_non_protected_attrib.get(feature_)])
                if len(value_total_variation_NDI_NII[data_non_protected_attrib.get(feature_)]) > 0:
                    total_variation_NDI_NII_mean_non_protected = np.sum(
                        value_total_variation_NDI_NII[data_non_protected_attrib.get(feature_)])
                if len(value_js_NDI_NII[data_non_protected_attrib.get(feature_)]) > 0:
                    JS_NDI_NII_mean_non_protected = np.sum(value_js_NDI_NII[data_non_protected_attrib.get(feature_)])
            feature_protected = data_protected_attrib.get(feature_)
            feature_non_protected = data_non_protected_attrib.get(feature_)

            feature_protected_str = data_protected_attrib_org.get(feature_)
            feature_non_protected_str = data_non_protected_attrib_org.get(feature_)
            data_writer.writerow(
                [feature_, swap_prop, '{}%'.format(swap_prop), 'Hellinger\n distance(+)', hellinger_NDI_NII_mean_protected - hellinger_NDI_NII_mean_non_protected,
                 np.std([hellinger_NDI_NII_mean_protected, hellinger_NDI_NII_mean_non_protected]), '{}\n({})'.format(feature_,feature_protected_str),
                 feature_non_protected_str])
            data_writer.writerow(
                [feature_, swap_prop, '{}%'.format(swap_prop), 'Wasserstein\n distance(+)',
                 wasserstein_NDI_NII_mean_protected-wasserstein_NDI_NII_mean_non_protected,
                 np.std([wasserstein_NDI_NII_mean_protected, wasserstein_NDI_NII_mean_non_protected]), '{}\n({})'.format(feature_,feature_protected_str),
                 feature_non_protected_str])

            data_writer.writerow(
                [key, swap_prop, '{}%'.format(swap_prop), 'Total variation\n distance(+)',
                 total_variation_NDI_NII_mean_protected-total_variation_NDI_NII_mean_non_protected,
                 np.std([total_variation_NDI_NII_mean_non_protected, total_variation_NDI_NII_mean_protected]), '{}\n({})'.format(feature_,feature_protected_str),
                 feature_non_protected_str])

            data_writer.writerow(
                [key, swap_prop, '{}%'.format(swap_prop), 'Jensen-Shannon\n divergence(+)',
                 JS_NDI_NII_mean_protected-JS_NDI_NII_mean_non_protected,
                 np.std([JS_NDI_NII_mean_non_protected, JS_NDI_NII_mean_protected]), '{}\n({})'.format(feature_,feature_protected_str),
                 feature_non_protected_str])


data_file.close()

