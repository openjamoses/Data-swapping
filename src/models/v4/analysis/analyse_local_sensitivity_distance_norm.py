import csv

import pandas as pd
import numpy as np
import os


#todo: group by tutorials: https://stackoverflow.com/questions/17679089/pandas-dataframe-groupby-two-columns-and-get-counts
perc_parity = 0.2
path = '../../dataset/'
#if __name__ == '__main__':
#df = pd.read_csv(path+'logging/log_KL_Divergence_overral_adult-45.csv')
#data_name = 'adult'
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


group_data = df_global.groupby(['ID','Feature','swap_proportion'])['Distortion_hellinger', 'Distortion_wasserstein', 'cramers_v_distortion', 'total_variation_distortion',
'Distortion_js', 'hellinger_div', 'wasserstein_div','cramers_v_div', 'total_variation_div','JS_Div', 'Casuality', 'Importance'].std().reset_index()
Feature = group_data.Feature.values.tolist()
swap_proportion = group_data.swap_proportion.values.tolist()
hellinger_div = group_data.hellinger_div.values.tolist()
wasserstein_div = group_data.wasserstein_div.values.tolist()
cramers_v_div = group_data.cramers_v_div.values.tolist()
total_variation_div = group_data.total_variation_div.values.tolist()
JS_Div = group_data.JS_Div.values.tolist()
Casuality = group_data.Casuality.values.tolist()
Importance = group_data.Importance.values.tolist()
data_dict_overal_std = {}
for i in range(len(Feature)):

    if Feature[i] in data_dict_overal_std.keys():
        if swap_proportion[i] in data_dict_overal_std[Feature[i]].keys():
            data_dict_overal_std[Feature[i]][swap_proportion[i]] = [hellinger_div[i], wasserstein_div[i], cramers_v_div[i], total_variation_div[i], JS_Div[i], Casuality[i], Importance[i]]
        else:
            data_dict_overal_std[Feature[i]][swap_proportion[i]] = [hellinger_div[i], wasserstein_div[i], cramers_v_div[i], total_variation_div[i], JS_Div[i], Casuality[i],
                                                                             Importance[i]]

    else:
        data_dict_overal_std[Feature[i]] = {}
        data_dict_overal_std[Feature[i]][swap_proportion[i]] = [hellinger_div[i], wasserstein_div[i], cramers_v_div[i], total_variation_div[i], JS_Div[i], Casuality[i],
                                                                         Importance[i]]
#todo: global
group_data = df_global.groupby(['ID', 'Feature','swap_proportion'])['distortion_feature','Distortion_hellinger', 'Distortion_wasserstein', 'cramers_v_distortion', 'total_variation_distortion',
'Distortion_js', 'hellinger_div', 'wasserstein_div','cramers_v_div','total_variation_div','JS_Div', 'Casuality', 'Importance'].mean().reset_index()
#print(len(group_data))
#print(group_data.head())
#group_data.to_csv(path + 'logging3/processed/KL_Divergence_local_adult_processed_global.csv', index=False)

data_file = open(path+path_output +'/processed'+'/Divergence_local_{}_processed_global_2_normalized.csv'.format(data_name), mode='w', newline='',
                 encoding='utf-8')
data_writer = csv.writer(data_file)
data_writer.writerow(['ID','Features','Swap', 'Swap_ratio', 'Distance_measure', 'Value', 'Normalized', 'Std'])
ID = group_data.ID.values.tolist()
Feature = group_data.Feature.values.tolist()
#Category = group_data.Category.values.tolist()
swap_proportion = group_data.swap_proportion.values.tolist()
#Distortion_hellinger = group_data.Distortion_hellinger.values.tolist()
#Distortion_wasserstein = group_data.Distortion_wasserstein.values.tolist()
#cramers_v_distortion = group_data.cramers_v_distortion.values.tolist()
#total_variation_distortion = group_data.total_variation_distortion.values.tolist()
distortion_feature = group_data.distortion_feature.values.tolist()

hellinger_div = group_data.hellinger_div.values.tolist()
wasserstein_div = group_data.wasserstein_div.values.tolist()
cramers_v_div = group_data.cramers_v_div.values.tolist()
total_variation_div = group_data.total_variation_div.values.tolist()
JS_Div = group_data.JS_Div.values.tolist()
for i in range(len(Feature)):
    swap_prop = int(swap_proportion[i]*100)
    hellinger_div_norm = hellinger_div[i]
    wasserstein_div_norm = wasserstein_div[i]
    cramers_v_div_norm = cramers_v_div[i]
    total_variation_div_norm = total_variation_div[i]
    JS_Div_norm = JS_Div[i]

    if (distortion_feature[i]) > 0:
        hellinger_div_norm = hellinger_div[i]/(distortion_feature[i])


        wasserstein_div_norm = wasserstein_div[i]/(distortion_feature[i])

        cramers_v_div_norm = cramers_v_div[i]/(distortion_feature[i])

        total_variation_div_norm = total_variation_div[i]/(distortion_feature[i])

        JS_Div_norm = JS_Div[i]/(distortion_feature[i])
    data_writer.writerow([ID[i], Feature[i], swap_proportion[i], '{}%'.format(swap_prop), 'Hellinger\n distance(+)', hellinger_div[i], hellinger_div_norm, data_dict_overal_std[Feature[i]][swap_proportion[i]][0]])
    data_writer.writerow(
        [ID[i], Feature[i],swap_proportion[i], '{}%'.format(swap_prop), 'Wasserstein\n distance(+)',
         wasserstein_div[i], wasserstein_div_norm, data_dict_overal_std[Feature[i]][swap_proportion[i]][1]])
    data_writer.writerow(
        [ID[i], Feature[i], swap_proportion[i], '{}%'.format(swap_prop), "Contingency\n coefficient(-)",
         cramers_v_div[i], cramers_v_div_norm, data_dict_overal_std[Feature[i]][swap_proportion[i]][2]])

    data_writer.writerow(
        [ID[i], Feature[i], swap_proportion[i], '{}%'.format(swap_prop), 'Total variation\n distance(+)',
         total_variation_div[i], total_variation_div_norm, data_dict_overal_std[Feature[i]][swap_proportion[i]][3]])

    data_writer.writerow(
        [ID[i], Feature[i],swap_proportion[i], '{}%'.format(swap_prop), 'Jensen???Shannon\n divergence(+)',
         JS_Div[i], JS_Div_norm, data_dict_overal_std[Feature[i]][swap_proportion[i]][4]])
data_file.close()







group_data = df_global.groupby(['ID','Feature','Category', 'swap_proportion'])['distortion_feature','Distortion_hellinger', 'Distortion_wasserstein', 'cramers_v_distortion', 'total_variation_distortion',
'Distortion_js', 'hellinger_div', 'wasserstein_div','cramers_v_div', 'total_variation_div','JS_Div', 'Casuality', 'Importance'].mean().reset_index()
print(group_data)
data_file = open(path+path_output +'/processed'+ '/Divergence_local_{}_processed_local_difference_normalized.csv'.format(data_name), mode='w', newline='',
                 encoding='utf-8')
data_writer = csv.writer(data_file)
data_writer.writerow(['Features','Swap', 'Swap_ratio', 'Distance_measure','Difference', 'Std', 'Protected', 'NonProtected'])


Feature = group_data.Feature.values.tolist()
Category = group_data.Category.values.tolist()
swap_proportion = group_data.swap_proportion.values.tolist()

#Distortion_hellinger = group_data.Distortion_hellinger.values.tolist()
#Distortion_wasserstein = group_data.Distortion_wasserstein.values.tolist()
#cramers_v_distortion = group_data.cramers_v_distortion.values.tolist()
#total_variation_distortion = group_data.total_variation_distortion.values.tolist()
distortion_feature = group_data.distortion_feature.values.tolist()

hellinger_div = group_data.hellinger_div.values.tolist()
wasserstein_div = group_data.wasserstein_div.values.tolist()
cramers_v_div = group_data.cramers_v_div.values.tolist()
total_variation_div = group_data.total_variation_div.values.tolist()
JS_Div = group_data.JS_Div.values.tolist()
Casuality = group_data.Casuality.values.tolist()
Importance = group_data.Importance.values.tolist()
data_dict_overal = {}
for i in range(len(Feature)):
    hellinger_div_norm = hellinger_div[i]
    wasserstein_div_norm = wasserstein_div[i]
    cramers_v_div_norm = cramers_v_div[i]
    total_variation_div_norm = total_variation_div[i]
    JS_Div_norm = JS_Div[i]

    if (distortion_feature[i]) > 0:
        hellinger_div_norm = hellinger_div[i] / (distortion_feature[i])

        wasserstein_div_norm = wasserstein_div[i] / (distortion_feature[i])

        total_variation_div_norm = total_variation_div[i] / (distortion_feature[i])

        JS_Div_norm = JS_Div[i] / (distortion_feature[i])

    if Feature[i] in data_dict_overal.keys():
        if swap_proportion[i] in data_dict_overal[Feature[i]].keys():
            data_dict_overal[Feature[i]][swap_proportion[i]][Category[i]] = [hellinger_div[i], wasserstein_div[i], cramers_v_div[i], total_variation_div[i], JS_Div[i], Casuality[i], Importance[i]]
        else:
            data_dict_overal[Feature[i]][swap_proportion[i]] = {}
            data_dict_overal[Feature[i]][swap_proportion[i]][Category[i]] = [hellinger_div[i], wasserstein_div[i], cramers_v_div[i], total_variation_div[i], JS_Div[i], Casuality[i],
                                                                             Importance[i]]

    else:
        data_dict_overal[Feature[i]] = {}
        data_dict_overal[Feature[i]][swap_proportion[i]] = {}
        data_dict_overal[Feature[i]][swap_proportion[i]][Category[i]] = [hellinger_div[i], wasserstein_div[i], cramers_v_div[i], total_variation_div[i], JS_Div[i], Casuality[i],
                                                                         Importance[i]]

for key, val in data_dict_overal.items():
    for key2, val2 in val.items():
        swap_prop = int(key2*100)
        val_protected, val_non_protected = None, None
        if data_protected_attrib.get(key) in val2.keys():
            val_protected = val2[data_protected_attrib.get(key)]
        if data_non_protected_attrib.get(key) in val2.keys():
            val_non_protected = val2[data_non_protected_attrib.get(key)]
        if val_protected != None and val_non_protected != None:
            data_writer.writerow([key, key2, '{}%'.format(swap_prop), 'Hellinger\n distance(+)', val_non_protected[0]-val_protected[0], np.std([val_protected[0], val_non_protected[0]]), data_protected_attrib.get(key), data_non_protected_attrib.get(key)])
            data_writer.writerow(
                [key, key2, '{}%'.format(swap_prop), 'Wasserstein\n distance(+)', val_non_protected[1]-val_protected[1],
                 np.std([val_protected[1], val_non_protected[1]]), data_protected_attrib.get(key), data_non_protected_attrib.get(key)])
            data_writer.writerow(
                [key, key2, '{}%'.format(swap_prop), "Contingency\n coefficient(-)", val_non_protected[2]-val_protected[2],
                 np.std([val_protected[2], val_non_protected[2]]),data_protected_attrib.get(key), data_non_protected_attrib.get(key)])

            data_writer.writerow(
                [key, key2, '{}%'.format(swap_prop), 'Total variation\n distance(+)', val_non_protected[3] - val_protected[3],
                 np.std([val_protected[3], val_non_protected[3]]), data_protected_attrib.get(key),
                 data_non_protected_attrib.get(key)])

            data_writer.writerow(
                [key, key2, '{}%'.format(swap_prop), 'Jensen-Shannon\n divergence(+)', val_non_protected[4]-val_protected[4],
                 np.std([val_protected[4], val_non_protected[4]]), data_protected_attrib.get(key), data_non_protected_attrib.get(key)])
        elif val_protected == None:

            data_writer.writerow(
                [key, key2, '{}%'.format(swap_prop), 'Hellinger\n distance(+)', val_non_protected[0],
                 np.std([0, val_non_protected[0]]), data_protected_attrib.get(key), data_non_protected_attrib.get(key)])
            data_writer.writerow(
                [key, key2, '{}%'.format(swap_prop), 'Wasserstein\n distance(+)', val_non_protected[1],
                 np.std([0, val_non_protected[1]]), data_protected_attrib.get(key), data_non_protected_attrib.get(key)])
            data_writer.writerow(
                [key, key2, '{}%'.format(swap_prop), "Contingency\n coefficient(-)", val_non_protected[2],
                 np.std([0, val_non_protected[2]]), data_protected_attrib.get(key), data_non_protected_attrib.get(key)])
            data_writer.writerow(
                [key, key2, '{}%'.format(swap_prop), 'Total variation\n distance(+)', val_non_protected[3],
                 np.std([0, val_non_protected[3]]), data_protected_attrib.get(key), data_non_protected_attrib.get(key)])

            data_writer.writerow(
                [key, key2, '{}%'.format(swap_prop), 'Jensen???Shannon\n divergence(+)', val_non_protected[4],
                 np.std([0, val_non_protected[4]]), data_protected_attrib.get(key), data_non_protected_attrib.get(key)])


        elif val_non_protected == None:

            data_writer.writerow(
                [key, key2, '{}%'.format(swap_prop), 'Hellinger\n distance(+)', 0 - val_protected[0],
                 np.std([val_protected[0], 0]), data_protected_attrib.get(key), data_non_protected_attrib.get(key)])
            data_writer.writerow(
                [key, key2, '{}%'.format(swap_prop), 'Wasserstein\n distance(+)', 0 - val_protected[1],
                 np.std([val_protected[1], 0]), data_protected_attrib.get(key), data_non_protected_attrib.get(key)])
            data_writer.writerow(
                [key, key2, '{}%'.format(swap_prop), "Contingency\n coefficient(-)", 0 - val_protected[2],
                 np.std([val_protected[2], 0]), data_protected_attrib.get(key), data_non_protected_attrib.get(key)])

            #print(val_protected, )
            data_writer.writerow(
                [key, key2, '{}%'.format(swap_prop), 'Total variation\n distance(+)', 0-val_protected[3],
                 np.std([0, val_protected[3]]), data_protected_attrib.get(key), data_non_protected_attrib.get(key)])

            data_writer.writerow(
                [key, key2, '{}%'.format(swap_prop), 'Jensen-Shannon\n divergence(+)', 0 - val_protected[4],
                 np.std([val_protected[4], 0]), data_protected_attrib.get(key), data_non_protected_attrib.get(key)])

data_file.close()
#print(group_data)
