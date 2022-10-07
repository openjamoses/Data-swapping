import csv

import pandas as pd
import numpy as np


#todo: group by tutorials: https://stackoverflow.com/questions/17679089/pandas-dataframe-groupby-two-columns-and-get-counts

path = '../../dataset/'
#if __name__ == '__main__':
#df = pd.read_csv(path+'logging/log_KL_Divergence_overral_adult-45.csv')
df_global = pd.read_csv(path + 'logging3/KL_Divergence_local_adult.csv')

print(len(df_global))
df_global = df_global.dropna(how='any')#axis='columns'
#print(df_global.columns.tolist())
#df_global = df_global.dropna(subset=['wasserstein_div'], how=-1)
df_global = df_global[df_global.wasserstein_div != -1]
print(len(df_global))
df_global = df_global[df_global.cramers_v_div != -1]
print(len(df_global))

#, 'swap_proportion'
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

group_data = df_global.groupby(['ID','Feature','Category', 'swap_proportion'])['Distortion_hellinger', 'Distortion_wasserstein', 'cramers_v_distortion',
'Distortion_js', 'hellinger_div', 'wasserstein_div','cramers_v_div','JS_Div', 'Casuality', 'Importance'].mean().reset_index()

data_file = open(path + 'logging3/processed/KL_Divergence_local_adult_processed_local_difference.csv', mode='w', newline='',
                 encoding='utf-8')
data_writer = csv.writer(data_file)
data_writer.writerow(['Features','Swap', 'Swap_ratio', 'Distance_measure','Difference', 'Std', 'Protected', 'NonProtected'])


Feature = group_data.Feature.values.tolist()
Category = group_data.Category.values.tolist()
swap_proportion = group_data.swap_proportion.values.tolist()

Distortion_hellinger = group_data.Distortion_hellinger.values.tolist()
Distortion_wasserstein = group_data.Distortion_wasserstein.values.tolist()
cramers_v_distortion = group_data.cramers_v_distortion.values.tolist()
Distortion_js = group_data.Distortion_js.values.tolist()

hellinger_div = group_data.hellinger_div.values.tolist()
wasserstein_div = group_data.wasserstein_div.values.tolist()
cramers_v_div = group_data.cramers_v_div.values.tolist()
JS_Div = group_data.JS_Div.values.tolist()
Casuality = group_data.Casuality.values.tolist()
Importance = group_data.Importance.values.tolist()
data_dict_overal = {}
for i in range(len(Feature)):
    if Feature[i] in data_dict_overal.keys():
        if swap_proportion[i] in data_dict_overal[Feature[i]].keys():
            data_dict_overal[Feature[i]][swap_proportion[i]][Category[i]] = [Distortion_hellinger[i], Distortion_wasserstein[i], cramers_v_distortion[i], Distortion_js[i],
                                                         hellinger_div[i], wasserstein_div[i], cramers_v_div[i], JS_Div[i], Casuality[i], Importance[i]]
        else:
            data_dict_overal[Feature[i]][swap_proportion[i]] = {}
            data_dict_overal[Feature[i]][swap_proportion[i]][Category[i]] = [Distortion_hellinger[i],
                                                                             Distortion_wasserstein[i],
                                                                             cramers_v_distortion[i], Distortion_js[i],
                                                                             hellinger_div[i], wasserstein_div[i],
                                                                             cramers_v_div[i], JS_Div[i], Casuality[i],
                                                                             Importance[i]]

    else:
        data_dict_overal[Feature[i]] = {}
        data_dict_overal[Feature[i]][swap_proportion[i]] = {}
        data_dict_overal[Feature[i]][swap_proportion[i]][Category[i]] = [Distortion_hellinger[i],
                                                                         Distortion_wasserstein[i],
                                                                         cramers_v_distortion[i], Distortion_js[i],
                                                                         hellinger_div[i], wasserstein_div[i],
                                                                         cramers_v_div[i], JS_Div[i], Casuality[i],
                                                                         Importance[i]]

for key, val in data_dict_overal.items():
    for key2, val2 in val.items():
        val_protected, val_non_protected = None, None
        if data_protected_attrib.get(key) in val2.keys():
            val_protected = val2[data_protected_attrib.get(key)]
        if data_non_protected_attrib.get(key) in val2.keys():
            val_non_protected = val2[data_non_protected_attrib.get(key)]
        if val_protected != None and val_non_protected != None:
            data_writer.writerow([key, key2, 'swap_{}'.format(key2), 'Hellinger', val_protected[4]-val_non_protected[4], np.std([val_protected[4], val_non_protected[4]]), data_protected_attrib.get(key), data_non_protected_attrib.get(key)])
            data_writer.writerow(
                [key, key2, 'swap_{}'.format(key2), 'Wasserstein', val_protected[5] - val_non_protected[5],
                 np.std([val_protected[5], val_non_protected[5]]), data_protected_attrib.get(key), data_non_protected_attrib.get(key)])
            data_writer.writerow(
                [key, key2, 'swap_{}'.format(key2), 'Cramers_v', val_protected[6] - val_non_protected[6],
                 np.std([val_protected[6], val_non_protected[6]]),data_protected_attrib.get(key), data_non_protected_attrib.get(key)])
            data_writer.writerow(
                [key, key2, 'swap_{}'.format(key2), 'JS Divergence', val_protected[7] - val_non_protected[7],
                 np.std([val_protected[7], val_non_protected[7]]), data_protected_attrib.get(key), data_non_protected_attrib.get(key)])
        elif val_protected == None:

            data_writer.writerow(
                [key, key2, 'swap_{}'.format(key2), 'Hellinger', 0 - val_non_protected[4],
                 np.std([0, val_non_protected[4]]), data_protected_attrib.get(key), data_non_protected_attrib.get(key)])
            data_writer.writerow(
                [key, key2, 'swap_{}'.format(key2), 'Wasserstein', 0 - val_non_protected[5],
                 np.std([0, val_non_protected[5]]), data_protected_attrib.get(key), data_non_protected_attrib.get(key)])
            data_writer.writerow(
                [key, key2, 'swap_{}'.format(key2), 'Cramers_v', 0 - val_non_protected[6],
                 np.std([0, val_non_protected[6]]), data_protected_attrib.get(key), data_non_protected_attrib.get(key)])
            data_writer.writerow(
                [key, key2, 'swap_{}'.format(key2), 'JS Divergence', 0 - val_non_protected[7],
                 np.std([0, val_non_protected[7]]), data_protected_attrib.get(key), data_non_protected_attrib.get(key)])

        elif val_non_protected == None:

            data_writer.writerow(
                [key, key2, 'swap_{}'.format(key2), 'Hellinger', 0 - val_protected[4],
                 np.std([val_protected[4], 0]), data_protected_attrib.get(key), data_non_protected_attrib.get(key)])
            data_writer.writerow(
                [key, key2, 'swap_{}'.format(key2), 'Wasserstein', 0 - val_protected[5],
                 np.std([val_protected[5], 0]), data_protected_attrib.get(key), data_non_protected_attrib.get(key)])
            data_writer.writerow(
                [key, key2, 'swap_{}'.format(key2), 'Cramers_v', 0 - val_protected[6],
                 np.std([val_protected[6], 0]), data_protected_attrib.get(key), data_non_protected_attrib.get(key)])
            data_writer.writerow(
                [key, key2, 'swap_{}'.format(key2), 'JS Divergence', 0 - val_protected[7],
                 np.std([val_protected[7], 0]), data_protected_attrib.get(key), data_non_protected_attrib.get(key)])

data_file.close()
#print(group_data)