import csv

import pandas as pd
import numpy as np
import os


#todo: group by tutorials: https://stackoverflow.com/questions/17679089/pandas-dataframe-groupby-two-columns-and-get-counts
alpha = 1.0
path = '../../dataset/'
#if __name__ == '__main__':
#df = pd.read_csv(path+'logging/log_KL_Divergence_overral_adult-45.csv')
#data_name = 'adult'
#data_name = 'clevelan_heart'
data_name = 'clevelan_heart-{}_'.format(alpha)  # _35_threshold
#data_name = 'Student'
path_output = 'logging3/{}/'.format(data_name)

swap_proportion_selected = 0.5
df_global = pd.read_csv(path + 'logging3/Divergence_2_columns_local_{}.csv'.format(data_name))
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

data_file = open(path+path_output +'/processed'+'/Confounding_Distance_features_{}.csv'.format(data_name), mode='w', newline='',
                 encoding='utf-8')
data_writer = csv.writer(data_file)
Feature = df_global.Feature.values.tolist()
data_writer.writerow(['Feature','Swap_proportion', 'Metric', 'Direct_Impact', 'Indirect_Impact', 'Impact_Sum','Impact_Product', 'Value'])


group_data = df_global.groupby(['Feature','Feature2','Category1', 'Category2', 'swap_proportion'])['distortion_multiple', 'distortion_feature', 'hellinger_div', 'wasserstein_div','cramers_v_div','total_variation_div','JS_Div', 'Casuality', 'Importance', 'effect_distance', 'effect_distance_contrained'].mean().reset_index()

data_file2 = open(path+path_output +'/processed'+'/Confounding_Others_features_{}_std.csv'.format(data_name), mode='w', newline='',
                 encoding='utf-8')
data_writer2 = csv.writer(data_file2)
data_writer2.writerow(['Feature','Swap_proportion', 'Metric', 'Direct_Impact', 'Indirect_Impact', 'Impact_Sum','Impact_Product', 'Value'])

Feature = group_data.Feature.values.tolist()
Feature2 = group_data.Feature2.values.tolist()

Category1 = group_data.Category1.values.tolist()
Category2 = group_data.Category2.values.tolist()

swap_proportion = group_data.swap_proportion.values.tolist()
hellinger_div = group_data.hellinger_div.values.tolist()
wasserstein_div = group_data.wasserstein_div.values.tolist()
cramers_v_div = group_data.cramers_v_div.values.tolist()
total_variation_div = group_data.total_variation_div.values.tolist()
JS_Div = group_data.JS_Div.values.tolist()
Casuality = group_data.Casuality.values.tolist()
effect_distance = group_data.effect_distance_contrained.values.tolist()
effect_distance_contrained = group_data.effect_distance.values.tolist()

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
            if swap_proportion[i] in data_direct_hellinger[Feature[i]].keys():
                if Category1[i] in data_direct_hellinger[Feature[i]][swap_proportion[i]].keys():
                    data_direct_hellinger[Feature[i]][swap_proportion[i]][Category1[i]].append(hellinger_div[i])
                    data_direct_wasserstein[Feature[i]][swap_proportion[i]][Category1[i]].append(wasserstein_div[i])
                    data_direct_cramers_v[Feature[i]][swap_proportion[i]][Category1[i]].append(cramers_v_div[i])
                    data_direct_total_variation[Feature[i]][swap_proportion[i]][Category1[i]].append(total_variation_div[i])
                    data_direct_JS[Feature[i]][swap_proportion[i]][Category1[i]].append(JS_Div[i])
                    data_direct_effect_distance[Feature[i]][swap_proportion[i]][Category1[i]].append(effect_distance[i])
                    data_direct_effect_distance_contrained[Feature[i]][swap_proportion[i]][Category1[i]].append(effect_distance_contrained[i])
                    data_direct_Casuality[Feature[i]][swap_proportion[i]][Category1[i]].append(Casuality[i])
                else:
                    data_direct_hellinger[Feature[i]][swap_proportion[i]][Category1[i]] = [hellinger_div[i]]
                    data_direct_wasserstein[Feature[i]][swap_proportion[i]][Category1[i]] = [wasserstein_div[i]]
                    data_direct_cramers_v[Feature[i]][swap_proportion[i]][Category1[i]] = [cramers_v_div[i]]
                    data_direct_total_variation[Feature[i]][swap_proportion[i]][Category1[i]] = [
                        total_variation_div[i]]
                    data_direct_JS[Feature[i]][swap_proportion[i]][Category1[i]] = [JS_Div[i]]
                    data_direct_effect_distance[Feature[i]][swap_proportion[i]][Category1[i]] = [effect_distance[i]]
                    data_direct_effect_distance_contrained[Feature[i]][swap_proportion[i]][Category1[i]] = [
                        effect_distance_contrained[i]]
                    data_direct_Casuality[Feature[i]][swap_proportion[i]][Category1[i]] = [Casuality[i]]
            else:
                data_direct_hellinger[Feature[i]][swap_proportion[i]] = {}
                data_direct_hellinger[Feature[i]][swap_proportion[i]][Category1[i]] = [hellinger_div[i]]
                # wasserstein
                data_direct_wasserstein[Feature[i]][swap_proportion[i]] = {}
                data_direct_wasserstein[Feature[i]][swap_proportion[i]][Category1[i]] = [wasserstein_div[i]]
                # cramers_v
                data_direct_cramers_v[Feature[i]][swap_proportion[i]] = {}
                data_direct_cramers_v[Feature[i]][swap_proportion[i]][Category1[i]] = [cramers_v_div[i]]
                # total_variation
                data_direct_total_variation[Feature[i]][swap_proportion[i]] = {}
                data_direct_total_variation[Feature[i]][swap_proportion[i]][Category1[i]] = [total_variation_div[i]]
                # JS
                data_direct_JS[Feature[i]][swap_proportion[i]] = {}
                data_direct_JS[Feature[i]][swap_proportion[i]][Category1[i]] = [JS_Div[i]]
                # effect_distance
                data_direct_effect_distance[Feature[i]][swap_proportion[i]] = {}
                data_direct_effect_distance[Feature[i]][swap_proportion[i]][Category1[i]] = [effect_distance[i]]
                # effect_distance_contrained
                data_direct_effect_distance_contrained[Feature[i]][swap_proportion[i]] = {}
                data_direct_effect_distance_contrained[Feature[i]][swap_proportion[i]][Category1[i]] = [effect_distance_contrained[i]]
                # Casuality
                data_direct_Casuality[Feature[i]][swap_proportion[i]] = {}
                data_direct_Casuality[Feature[i]][swap_proportion[i]][Category1[i]] = [Casuality[i]]
                
        else:
            data_direct_hellinger[Feature[i]] = {}
            data_direct_hellinger[Feature[i]][swap_proportion[i]] = {}
            data_direct_hellinger[Feature[i]][swap_proportion[i]][Category1[i]] = [hellinger_div[i]]
            # wasserstein
            data_direct_wasserstein[Feature[i]] = {}
            data_direct_wasserstein[Feature[i]][swap_proportion[i]] = {}
            data_direct_wasserstein[Feature[i]][swap_proportion[i]][Category1[i]] = [wasserstein_div[i]]
            # cramers_v
            data_direct_cramers_v[Feature[i]] = {}
            data_direct_cramers_v[Feature[i]][swap_proportion[i]] = {}
            data_direct_cramers_v[Feature[i]][swap_proportion[i]][Category1[i]] = [cramers_v_div[i]]
            # total_variation
            data_direct_total_variation[Feature[i]] = {}
            data_direct_total_variation[Feature[i]][swap_proportion[i]] = {}
            data_direct_total_variation[Feature[i]][swap_proportion[i]][Category1[i]] = [total_variation_div[i]]
            # JS
            data_direct_JS[Feature[i]] = {}
            data_direct_JS[Feature[i]][swap_proportion[i]] = {}
            data_direct_JS[Feature[i]][swap_proportion[i]][Category1[i]] = [JS_Div[i]]
            # effect_distance
            data_direct_effect_distance[Feature[i]] = {}
            data_direct_effect_distance[Feature[i]][swap_proportion[i]] = {}
            data_direct_effect_distance[Feature[i]][swap_proportion[i]][Category1[i]] = [effect_distance[i]]
            # effect_distance_contrained
            data_direct_effect_distance_contrained[Feature[i]] = {}
            data_direct_effect_distance_contrained[Feature[i]][swap_proportion[i]] = {}
            data_direct_effect_distance_contrained[Feature[i]][swap_proportion[i]][Category1[i]] = [
                effect_distance_contrained[i]]
            # Casuality
            data_direct_Casuality[Feature[i]] = {}
            data_direct_Casuality[Feature[i]][swap_proportion[i]] = {}
            data_direct_Casuality[Feature[i]][swap_proportion[i]][Category1[i]] = [Casuality[i]]
    else:
        type_ = 'Indirect'
        #TODO: Indireect casual effect starts here....
        if Feature[i] in data_indirect_hellinger.keys():
            if swap_proportion[i] in data_indirect_hellinger[Feature[i]].keys():
                if Category1[i] in data_indirect_hellinger[Feature[i]][swap_proportion[i]].keys():
                    data_indirect_hellinger[Feature[i]][swap_proportion[i]][Category1[i]].append(hellinger_div[i])
                    data_indirect_wasserstein[Feature[i]][swap_proportion[i]][Category1[i]].append(wasserstein_div[i])
                    data_indirect_cramers_v[Feature[i]][swap_proportion[i]][Category1[i]].append(cramers_v_div[i])
                    data_indirect_total_variation[Feature[i]][swap_proportion[i]][Category1[i]].append(
                        total_variation_div[i])
                    data_indirect_JS[Feature[i]][swap_proportion[i]][Category1[i]].append(JS_Div[i])
                    data_indirect_effect_distance[Feature[i]][swap_proportion[i]][Category1[i]].append(effect_distance[i])
                    data_indirect_effect_distance_contrained[Feature[i]][swap_proportion[i]][Category1[i]].append(
                        effect_distance_contrained[i])
                    data_indirect_Casuality[Feature[i]][swap_proportion[i]][Category1[i]].append(Casuality[i])
                else:
                    data_indirect_hellinger[Feature[i]][swap_proportion[i]][Category1[i]] = [hellinger_div[i]]
                    data_indirect_wasserstein[Feature[i]][swap_proportion[i]][Category1[i]] = [wasserstein_div[i]]
                    data_indirect_cramers_v[Feature[i]][swap_proportion[i]][Category1[i]] = [cramers_v_div[i]]
                    data_indirect_total_variation[Feature[i]][swap_proportion[i]][Category1[i]] = [
                        total_variation_div[i]]
                    data_indirect_JS[Feature[i]][swap_proportion[i]][Category1[i]] = [JS_Div[i]]
                    data_indirect_effect_distance[Feature[i]][swap_proportion[i]][Category1[i]] = [effect_distance[i]]
                    data_indirect_effect_distance_contrained[Feature[i]][swap_proportion[i]][Category1[i]] = [
                        effect_distance_contrained[i]]
                    data_indirect_Casuality[Feature[i]][swap_proportion[i]][Category1[i]] = [Casuality[i]]
            else:
                data_indirect_hellinger[Feature[i]][swap_proportion[i]] = {}
                data_indirect_hellinger[Feature[i]][swap_proportion[i]][Category1[i]] = [hellinger_div[i]]
                # wasserstein
                data_indirect_wasserstein[Feature[i]][swap_proportion[i]] = {}
                data_indirect_wasserstein[Feature[i]][swap_proportion[i]][Category1[i]] = [wasserstein_div[i]]
                # cramers_v
                data_indirect_cramers_v[Feature[i]][swap_proportion[i]] = {}
                data_indirect_cramers_v[Feature[i]][swap_proportion[i]][Category1[i]] = [cramers_v_div[i]]
                # total_variation
                data_indirect_total_variation[Feature[i]][swap_proportion[i]] = {}
                data_indirect_total_variation[Feature[i]][swap_proportion[i]][Category1[i]] = [total_variation_div[i]]
                # JS
                data_indirect_JS[Feature[i]][swap_proportion[i]] = {}
                data_indirect_JS[Feature[i]][swap_proportion[i]][Category1[i]] = [JS_Div[i]]
                # effect_distance
                data_indirect_effect_distance[Feature[i]][swap_proportion[i]] = {}
                data_indirect_effect_distance[Feature[i]][swap_proportion[i]][Category1[i]] = [effect_distance[i]]
                # effect_distance_contrained
                data_indirect_effect_distance_contrained[Feature[i]][swap_proportion[i]] = {}
                data_indirect_effect_distance_contrained[Feature[i]][swap_proportion[i]][Category1[i]] = [
                    effect_distance_contrained[i]]
                # Casuality
                data_indirect_Casuality[Feature[i]][swap_proportion[i]] = {}
                data_indirect_Casuality[Feature[i]][swap_proportion[i]][Category1[i]] = [Casuality[i]]

        else:
            data_indirect_hellinger[Feature[i]] = {}
            data_indirect_hellinger[Feature[i]][swap_proportion[i]] = {}
            data_indirect_hellinger[Feature[i]][swap_proportion[i]][Category1[i]] = [hellinger_div[i]]
            # wasserstein
            data_indirect_wasserstein[Feature[i]] = {}
            data_indirect_wasserstein[Feature[i]][swap_proportion[i]] = {}
            data_indirect_wasserstein[Feature[i]][swap_proportion[i]][Category1[i]] = [wasserstein_div[i]]
            # cramers_v
            data_indirect_cramers_v[Feature[i]] = {}
            data_indirect_cramers_v[Feature[i]][swap_proportion[i]] = {}
            data_indirect_cramers_v[Feature[i]][swap_proportion[i]][Category1[i]] = [cramers_v_div[i]]
            # total_variation
            data_indirect_total_variation[Feature[i]] = {}
            data_indirect_total_variation[Feature[i]][swap_proportion[i]] = {}
            data_indirect_total_variation[Feature[i]][swap_proportion[i]][Category1[i]] = [total_variation_div[i]]
            # JS
            data_indirect_JS[Feature[i]] = {}
            data_indirect_JS[Feature[i]][swap_proportion[i]] = {}
            data_indirect_JS[Feature[i]][swap_proportion[i]][Category1[i]] = [JS_Div[i]]
            # effect_distance
            data_indirect_effect_distance[Feature[i]] = {}
            data_indirect_effect_distance[Feature[i]][swap_proportion[i]] = {}
            data_indirect_effect_distance[Feature[i]][swap_proportion[i]][Category1[i]] = [effect_distance[i]]
            # effect_distance_contrained
            data_indirect_effect_distance_contrained[Feature[i]] = {}
            data_indirect_effect_distance_contrained[Feature[i]][swap_proportion[i]] = {}
            data_indirect_effect_distance_contrained[Feature[i]][swap_proportion[i]][Category1[i]] = [
                effect_distance_contrained[i]]
            # Casuality
            data_indirect_Casuality[Feature[i]] = {}
            data_indirect_Casuality[Feature[i]][swap_proportion[i]] = {}
            data_indirect_Casuality[Feature[i]][swap_proportion[i]][Category1[i]] = [Casuality[i]]
            

for feature_, val in data_direct_hellinger.items():
    for swap_, val2 in val.items():
        keys_ = list(val2.keys())
        swap_prop = int(swap_ * 100)
        #todo: direct impact results 1
        direct_effect_hellinger = abs(np.mean(val2.get(keys_[0])) - np.mean(val2.get(keys_[1])))
        direct_effect_wasserstein = abs(np.mean(data_direct_wasserstein[feature_][swap_][keys_[0]]) - np.mean(data_direct_wasserstein[feature_][swap_][keys_[1]]))
        direct_effect_cramers_v = abs(np.mean(data_direct_cramers_v[feature_][swap_][keys_[0]]) - np.mean(
            data_direct_cramers_v[feature_][swap_][keys_[1]]))
        direct_effect_total_variation = abs(np.mean(data_direct_total_variation[feature_][swap_][keys_[0]]) - np.mean(
            data_direct_total_variation[feature_][swap_][keys_[1]]))
        direct_effect_JS = abs(np.mean(data_direct_JS[feature_][swap_][keys_[0]]) - np.mean(
            data_direct_JS[feature_][swap_][keys_[1]]))
        direct_effect_Casuality = abs(np.mean(data_direct_Casuality[feature_][swap_][keys_[0]]) - np.mean(
            data_direct_Casuality[feature_][swap_][keys_[1]]))
        direct_effect_distance = abs(np.mean(data_direct_effect_distance[feature_][swap_][keys_[0]]) - np.mean(
            data_direct_effect_distance[feature_][swap_][keys_[1]]))
        direct_effect_distance_contrained = abs(np.mean(data_direct_effect_distance_contrained[feature_][swap_][keys_[0]]) - np.mean(
            data_direct_effect_distance_contrained[feature_][swap_][keys_[1]]))

        # todo: In direct impact results 1
        indirect_effect_hellinger = abs(np.mean(data_indirect_hellinger[feature_][swap_][keys_[0]]) - np.mean(data_indirect_hellinger[feature_][swap_][keys_[1]]))
        indirect_effect_wasserstein = abs(np.mean(data_indirect_wasserstein[feature_][swap_][keys_[0]]) - np.mean(
            data_indirect_wasserstein[feature_][swap_][keys_[1]]))
        indirect_effect_cramers_v = abs(np.mean(data_indirect_cramers_v[feature_][swap_][keys_[0]]) - np.mean(
            data_indirect_cramers_v[feature_][swap_][keys_[1]]))
        indirect_effect_total_variation = abs(np.mean(data_indirect_total_variation[feature_][swap_][keys_[0]]) - np.mean(
            data_indirect_total_variation[feature_][swap_][keys_[1]]))
        #print(data_indirect_JS[feature_][swap_][keys_[0]], data_indirect_JS[feature_][swap_][keys_[1]])
        indirect_effect_JS = abs(np.mean(data_indirect_JS[feature_][swap_][keys_[0]]) - np.mean(
            data_indirect_JS[feature_][swap_][keys_[1]]))
        #print(data_indirect_Casuality[feature_][swap_][keys_[0]], data_indirect_Casuality[feature_][swap_][keys_[1]])
        indirect_effect_Casuality = abs(np.mean(data_indirect_Casuality[feature_][swap_][keys_[0]]) - np.mean(
            data_indirect_Casuality[feature_][swap_][keys_[1]]))
        indirect_effect_distance = abs(np.mean(data_indirect_effect_distance[feature_][swap_][keys_[0]]) - np.mean(
            data_indirect_effect_distance[feature_][swap_][keys_[1]]))
        indirect_effect_distance_contrained = abs(
            np.mean(data_indirect_effect_distance_contrained[feature_][swap_][keys_[0]]) - np.mean(
                data_indirect_effect_distance_contrained[feature_][swap_][keys_[1]]))

        hellinger_values = [i for i in val2[keys_[0]]]
        hellinger_values.extend(val2[keys_[1]])
        hellinger_values.extend(data_indirect_hellinger[feature_][swap_][keys_[0]])
        hellinger_values.extend(data_indirect_hellinger[feature_][swap_][keys_[1]])

        wasserstein_values = [i for i in data_direct_wasserstein[feature_][swap_][keys_[0]]]
        wasserstein_values.extend(data_direct_wasserstein[feature_][swap_][keys_[1]])
        wasserstein_values.extend(data_indirect_wasserstein[feature_][swap_][keys_[0]])
        wasserstein_values.extend(data_indirect_wasserstein[feature_][swap_][keys_[1]])

        cramers_v_values = [i for i in data_direct_cramers_v[feature_][swap_][keys_[0]]]
        cramers_v_values.extend(data_direct_cramers_v[feature_][swap_][keys_[1]])
        cramers_v_values.extend(data_indirect_cramers_v[feature_][swap_][keys_[0]])
        cramers_v_values.extend(data_indirect_cramers_v[feature_][swap_][keys_[1]])

        total_variation_values = [i for i in data_direct_total_variation[feature_][swap_][keys_[0]]]
        total_variation_values.extend(data_direct_total_variation[feature_][swap_][keys_[1]])
        total_variation_values.extend(data_indirect_total_variation[feature_][swap_][keys_[0]])
        total_variation_values.extend(data_indirect_total_variation[feature_][swap_][keys_[1]])

        JS_values = [i for i in data_direct_JS[feature_][swap_][keys_[0]]]
        JS_values.extend(data_direct_JS[feature_][swap_][keys_[1]])
        JS_values.extend(data_indirect_JS[feature_][swap_][keys_[0]])
        JS_values.extend(data_indirect_JS[feature_][swap_][keys_[1]])

        Casuality_values = [i for i in data_direct_Casuality[feature_][swap_][keys_[0]]]
        Casuality_values.extend(data_direct_Casuality[feature_][swap_][keys_[1]])
        Casuality_values.extend(data_indirect_Casuality[feature_][swap_][keys_[0]])
        Casuality_values.extend(data_indirect_Casuality[feature_][swap_][keys_[1]])

        effect_distance_values = [i for i in data_direct_effect_distance[feature_][swap_][keys_[0]]]
        effect_distance_values.extend(data_direct_effect_distance[feature_][swap_][keys_[1]])
        effect_distance_values.extend(data_indirect_effect_distance[feature_][swap_][keys_[0]])
        effect_distance_values.extend(data_indirect_effect_distance[feature_][swap_][keys_[1]])

        effect_distance_contrained_values = [i for i in data_direct_effect_distance_contrained[feature_][swap_][keys_[0]]]
        effect_distance_contrained_values.extend(data_direct_effect_distance_contrained[feature_][swap_][keys_[1]])
        effect_distance_contrained_values.extend(data_indirect_effect_distance_contrained[feature_][swap_][keys_[0]])
        effect_distance_contrained_values.extend(data_indirect_effect_distance_contrained[feature_][swap_][keys_[1]])

        data_writer.writerow(
            [feature_, '{}%'.format(swap_prop), 'Hellinger\n distance(+)', direct_effect_hellinger, indirect_effect_hellinger, direct_effect_hellinger+indirect_effect_hellinger, direct_effect_hellinger * indirect_effect_hellinger,
             np.mean(hellinger_values)])
        data_writer.writerow(
            [feature_, '{}%'.format(swap_prop), 'Wasserstein\n distance(+)', direct_effect_wasserstein,
             indirect_effect_wasserstein, direct_effect_wasserstein + indirect_effect_wasserstein,
             direct_effect_wasserstein * indirect_effect_wasserstein,
             np.mean(wasserstein_values)])
        data_writer.writerow(
            [feature_, '{}%'.format(swap_prop), "Contingency\n coefficient(-)", direct_effect_cramers_v,
             indirect_effect_cramers_v, direct_effect_cramers_v + indirect_effect_cramers_v,
             direct_effect_cramers_v * indirect_effect_cramers_v,
             np.mean(cramers_v_values)])
        data_writer.writerow(
            [feature_, '{}%'.format(swap_prop), 'Total variation\n distance(+)', direct_effect_total_variation,
             indirect_effect_total_variation, direct_effect_total_variation + indirect_effect_total_variation,
             direct_effect_total_variation * indirect_effect_total_variation,
             np.mean(total_variation_values)])
        data_writer.writerow(
            [feature_, '{}%'.format(swap_prop), 'Jensen-Shannon\n divergence(+)', direct_effect_JS,
             indirect_effect_JS, direct_effect_JS + indirect_effect_JS,
             direct_effect_JS * indirect_effect_JS,
             np.mean(JS_values)])


        ##TODO; Other casual metrics
        data_writer2.writerow(
            [feature_, '{}%'.format(swap_prop), "Contingency\n coefficient(-)", direct_effect_Casuality,
             indirect_effect_Casuality, direct_effect_Casuality + indirect_effect_Casuality,
             direct_effect_Casuality * indirect_effect_Casuality,
             np.mean(Casuality_values)])
        data_writer2.writerow(
            [feature_, '{}%'.format(swap_prop), 'Impact\n distance(+)', direct_effect_distance,
             indirect_effect_distance, direct_effect_distance + indirect_effect_distance,
             direct_effect_distance * indirect_effect_distance,
             np.mean(effect_distance_values)])
        data_writer2.writerow(
            [feature_, '{}%'.format(swap_prop), 'Impact \n Distance Contrained(+)', direct_effect_distance_contrained,
             indirect_effect_distance_contrained, direct_effect_distance_contrained + indirect_effect_distance_contrained,
             direct_effect_distance_contrained * indirect_effect_distance_contrained,
             np.mean(effect_distance_contrained_values)])

data_file.close()
data_file2.close()

