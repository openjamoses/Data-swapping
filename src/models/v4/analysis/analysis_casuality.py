import csv

import pandas as pd
import numpy as np
import os


#todo: group by tutorials: https://stackoverflow.com/questions/17679089/pandas-dataframe-groupby-two-columns-and-get-counts
perc_parity = 0.2
path = '../../dataset/'
#if __name__ == '__main__':
#data_name = 'adult'
#data_name = 'clevelan_heart'
data_name = 'adult-{}_'.format(perc_parity)
#data_name = 'Student'
#data_name = 'compas'
path_output = 'logging3/{}/'.format(data_name)
df_global = pd.read_csv(path + 'logging3/KL_Divergence_local_{}.csv'.format(data_name))
if not os.path.exists(path_output + data_name):
    os.makedirs(path_output + data_name)
if not os.path.exists(path+path_output+'/processed'):
    os.makedirs(path+path_output +'/processed')

print(len(df_global))
df_global = df_global.dropna(how='any')#axis='columns'
#print(df_global.columns.tolist())
#df_global = df_global.dropna(subset=['wasserstein_div'], how=-1)
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


#'swap_proportion'
#todo: local
group_data = df_global.groupby(['ID','Feature','Category'])['Distortion_hellinger', 'Distortion_wasserstein', 'cramers_v_distortion',
'Distortion_js', 'hellinger_div', 'wasserstein_div','cramers_v_div','JS_Div', 'Casuality', 'Importance'].mean().reset_index()
#group_data.to_csv(path + 'logging3/processed/KL_Divergence_local_adult_processed_local_analysis.csv', index=False)

Feature = group_data.Feature.values.tolist()
Category = group_data.Category.values.tolist()
Importance = group_data.Importance.values.tolist()
data_dict = {}
data_dict_keys = {}
for i in range(len(Feature)):
    if Feature[i] in data_dict.keys():
        data_dict[Feature[i]].append(Importance[i])
        data_dict_keys[Feature[i]].append(Category[i])
    else:
        data_dict[Feature[i]] = [Importance[i]]
        data_dict_keys[Feature[i]] = [Category[i]]
data_protected_attrib = {}
data_non_protected_attrib = {}
for key, val in data_dict.items():
    if val[0] > val[1]:
        data_protected_attrib[key] = data_dict_keys[key][1]
        data_non_protected_attrib[key] = data_dict_keys[key][0]
    else:
        data_protected_attrib[key] = data_dict_keys[key][0]
        data_non_protected_attrib[key] = data_dict_keys[key][1]

#todo: global
group_data = df_global.groupby(['ID', 'Feature','swap_proportion'])['Casuality', 'Importance'].mean().reset_index()
#print(len(group_data))
#print(group_data.head())
#group_data.to_csv(path + 'logging3/processed/KL_Divergence_local_adult_processed_global.csv', index=False)
data_file = open(path+path_output +'/processed'+'/Casuality_local_adult_processed_global_casuality.csv', mode='w', newline='',
                 encoding='utf-8')
data_writer = csv.writer(data_file)
data_writer.writerow(['ID','Features','Swap', 'Swap_ratio', 'Casuality_metrics', 'Value'])
ID = group_data.ID.values.tolist()
Feature = group_data.Feature.values.tolist()
swap_proportion = group_data.swap_proportion.values.tolist()
Casuality = group_data.Casuality.values.tolist()
Importance = group_data.Importance.values.tolist()
for i in range(len(Feature)):

    swap_prop = int(swap_proportion[i]*100)
    #data_writer.writerow(
    #    [ID[i], Feature[i],swap_proportion[i], '{}%'.format(swap_prop), 'Permutation\n importance(+)',
    #     Importance[i]])
    data_writer.writerow(
        [ID[i], Feature[i], swap_proportion[i], '{}%'.format(swap_prop), 'Propensity\n score(+)',
         Casuality[i]])
data_file.close()

# todo: local analysis

group_data = df_global.groupby(['ID','Feature','Category', 'swap_proportion'])['Distortion_hellinger', 'Distortion_wasserstein', 'cramers_v_distortion',
'Distortion_js', 'hellinger_div', 'wasserstein_div','cramers_v_div','JS_Div', 'Casuality', 'Importance'].mean().reset_index()

data_file = open(path+path_output +'/processed'+'/Casuality_local_adult_processed_local_difference.csv', mode='w', newline='',
                 encoding='utf-8')
data_writer = csv.writer(data_file)
data_writer.writerow(['Features','Swap', 'Swap_ratio', 'Distance_measure','Difference', 'Std', 'Protected', 'NonProtected'])


Feature = group_data.Feature.values.tolist()
Category = group_data.Category.values.tolist()
swap_proportion = group_data.swap_proportion.values.tolist()

Casuality = group_data.Casuality.values.tolist()
Importance = group_data.Importance.values.tolist()
data_dict_overal = {}
for i in range(len(Feature)):
    if Feature[i] in data_dict_overal.keys():
        if swap_proportion[i] in data_dict_overal[Feature[i]].keys():
            data_dict_overal[Feature[i]][swap_proportion[i]][Category[i]] = [Casuality[i], Importance[i]]
        else:
            data_dict_overal[Feature[i]][swap_proportion[i]] = {}
            data_dict_overal[Feature[i]][swap_proportion[i]][Category[i]] = [Casuality[i],
                                                                             Importance[i]]

    else:
        data_dict_overal[Feature[i]] = {}
        data_dict_overal[Feature[i]][swap_proportion[i]] = {}
        data_dict_overal[Feature[i]][swap_proportion[i]][Category[i]] = [Casuality[i],
                                                                         Importance[i]]

for key, val in data_dict_overal.items():
    #print(key)

    for key2, val2 in val.items():
        swap_prop = int(key2 * 100)
        val_protected, val_non_protected = None, None
        if data_protected_attrib.get(key) in val2.keys():
            val_protected = val2[data_protected_attrib.get(key)]
        if data_non_protected_attrib.get(key) in val2.keys():
            val_non_protected = val2[data_non_protected_attrib.get(key)]
        if val_protected != None and val_non_protected != None:
            #data_writer.writerow(
            #    [key, key2, '{}%'.format(swap_prop), 'Permutation\n importance(+)', val_non_protected[1]-val_protected[1],
            #     np.std([val_protected[1], val_non_protected[1]]), '{}\n[{}]'.format(key, data_protected_attrib.get(key)), data_non_protected_attrib.get(key)])
            data_writer.writerow(
                [key, key2, '{}%'.format(swap_prop), 'Propensity\n score(+)', val_non_protected[0]-val_protected[0],
                 np.std([val_protected[0], val_non_protected[0]]), '{}\n[{}]'.format(key, data_protected_attrib.get(key)), data_non_protected_attrib.get(key)])
        elif val_protected == None:

            #data_writer.writerow(
            #    [key, key2, '{}%'.format(swap_prop), 'Permutation\n importance(+)', val_non_protected[1],
            #     np.std([0, val_non_protected[1]]), '{}\n[{}]'.format(key, data_protected_attrib.get(key)), data_non_protected_attrib.get(key)])
            data_writer.writerow(
                [key, key2, '{}%'.format(swap_prop), 'Propensity\n score(+)', val_non_protected[0],
                 np.std([0, val_non_protected[0]]), '{}\n[{}]'.format(key, data_protected_attrib.get(key)), data_non_protected_attrib.get(key)])

        elif val_non_protected == None:
            #<=
            #data_writer.writerow(
            #    [key, key2, '{}%'.format(swap_prop), 'Permutation\n importance(+)', 0 - val_protected[1],
            #     np.std([val_protected[1], 0]), '{}\n[{}]'.format(key, data_protected_attrib.get(key)), data_non_protected_attrib.get(key)])
            data_writer.writerow(
                [key, key2, '{}%'.format(swap_prop), 'Propensity\n score(+)', 0 - val_protected[0],
                 np.std([val_protected[0], 0]), '{}\n[{}]'.format(key, data_protected_attrib.get(key)), data_non_protected_attrib.get(key)])
data_file.close()
#print(group_data)
