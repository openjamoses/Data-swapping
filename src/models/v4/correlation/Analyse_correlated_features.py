import csv

import pandas as pd
import numpy as np
import os


#todo: group by tutorials: https://stackoverflow.com/questions/17679089/pandas-dataframe-groupby-two-columns-and-get-counts

path = '../../dataset/'
#if __name__ == '__main__':
#df = pd.read_csv(path+'logging/log_KL_Divergence_overral_adult-45.csv')
data_name = 'adult'
#data_name = 'clevelan_heart'
#data_name = 'Student'
path_output = 'logging3/{}/'.format(data_name)

swap_proportion_selected = 0.5
df_global = pd.read_csv(path + 'logging3/Divergence_2_columns_{}.csv'.format(data_name))
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

data_file = open(path+path_output +'/processed'+'/Correlated_features_{}.csv'.format(data_name), mode='w', newline='',
                 encoding='utf-8')
data_writer = csv.writer(data_file)
Feature = df_global.Feature.values.tolist()
row = ['Swap_ration','Measurement', 'Feature']
feature_set = set(Feature)
for feature_ in feature_set:
    row.append(feature_)
data_writer.writerow(row)

data_file2 = open(path+path_output +'/processed'+'/Correlated_features_{}_std.csv'.format(data_name), mode='w', newline='',
                 encoding='utf-8')
data_writer2 = csv.writer(data_file2)
row = ['Swap_ration','Measurement', 'Feature']
for feature_ in feature_set:
    row.append(feature_)
data_writer2.writerow(row)

Feature2 = df_global.Feature2.values.tolist()
swap_proportion = df_global.swap_proportion.values.tolist()
hellinger_div = df_global.hellinger_div.values.tolist()
wasserstein_div = df_global.wasserstein_div.values.tolist()
cramers_v_div = df_global.cramers_v_div.values.tolist()
total_variation_div = df_global.total_variation_div.values.tolist()
JS_Div = df_global.JS_Div.values.tolist()
Casuality = df_global.Casuality.values.tolist()

for feature_ in feature_set:
    row_hellinger = [swap_proportion_selected, 'Hellinger distance', feature_]
    row_wasserstein = [swap_proportion_selected, 'Wasserstein distance', feature_]
    row_cramers_v = [swap_proportion_selected, 'Contingency coefficient', feature_]
    row_total_variation = [swap_proportion_selected, 'Total variation distance', feature_]
    row_js = [swap_proportion_selected, 'Jensen–Shannon divergence',feature_]
    row_Casuality = [swap_proportion_selected, 'Propensity score', feature_]
    row_Combined = [swap_proportion_selected, 'Combined score', feature_]

    # rows with std
    row_hellinger2 = [swap_proportion_selected, 'Hellinger distance', feature_]
    row_wasserstein2 = [swap_proportion_selected, 'Wasserstein distance', feature_]
    row_cramers_v2 = [swap_proportion_selected, 'Contingency coefficient', feature_]
    row_total_variation2 = [swap_proportion_selected, 'Total variation distance', feature_]
    row_js2 = [swap_proportion_selected, 'Jensen–Shannon divergence', feature_]
    row_Casuality2 = [swap_proportion_selected, 'Propensity score', feature_]
    row_Combined2 = [swap_proportion_selected, 'Combined score', feature_]

    for feature_2 in feature_set:
        val_correlation = 1
        if feature_2 != feature_:
            correlated_hellinger_div = []
            correlated_wasserstein_div = []
            correlated_cramers_v_div = []
            correlated_total_variation_div = []
            correlated_JS_Div = []
            correlated_Casuality = []
            correlated_Combined = []
            for i in range(len(Feature)):
                div_crammer_overral = hellinger_div[i] * wasserstein_div[i] * total_variation_div[i] * JS_Div[i] * \
                                          Casuality[i]
                if cramers_v_div[i] != 0:
                    div_crammer_overral = hellinger_div[i] * wasserstein_div[i] * total_variation_div[i] * JS_Div[i] * \
                                          Casuality[i] / cramers_v_div[i]

                if swap_proportion[i] == swap_proportion_selected and Feature[i] == feature_ and Feature2[i] == feature_2:
                    correlated_hellinger_div.append(hellinger_div[i])
                    correlated_wasserstein_div.append(wasserstein_div[i])
                    correlated_cramers_v_div.append(cramers_v_div[i])
                    correlated_total_variation_div.append(total_variation_div[i])
                    correlated_JS_Div.append(JS_Div[i])
                    correlated_Casuality.append(Casuality[i])
                    correlated_Combined.append(div_crammer_overral)
            row_hellinger.append(round(np.mean(correlated_hellinger_div),3))
            row_wasserstein.append(round(np.mean(correlated_wasserstein_div),3))
            row_cramers_v.append(round(np.mean(correlated_cramers_v_div),3))
            row_total_variation.append(round(np.mean(correlated_total_variation_div),3))
            row_js.append(round(np.mean(correlated_JS_Div),3))
            row_Casuality.append(round(np.mean(correlated_Casuality),3))
            row_Combined.append(round(np.mean(correlated_Combined), 4))

            # rows with std
            row_hellinger2.append('{}({})'.format(np.mean(correlated_hellinger_div), np.std(correlated_hellinger_div)))
            row_wasserstein2.append('{}({})'.format(np.mean(correlated_wasserstein_div), np.std(correlated_wasserstein_div)))
            row_cramers_v2.append('{}({})'.format(np.mean(correlated_cramers_v_div), np.std(correlated_cramers_v_div)))
            row_total_variation2.append('{}({})'.format(np.mean(correlated_total_variation_div), np.std(correlated_total_variation_div)))
            row_js2.append('{}({})'.format(np.mean(correlated_JS_Div), np.std(correlated_JS_Div)))
            row_Casuality2.append('{}({})'.format(np.mean(correlated_Casuality), np.std(correlated_Casuality)))
            row_Combined2.append('{}({})'.format(round(np.mean(correlated_Combined),4), round(np.std(correlated_Combined),4)))
        else:
            row_hellinger.append(val_correlation)
            row_wasserstein.append(val_correlation)
            row_cramers_v.append(val_correlation)
            row_total_variation.append(val_correlation)
            row_js.append(val_correlation)
            row_Casuality.append(val_correlation)
            row_Combined.append(val_correlation)

            # rows with std
            row_hellinger2.append(val_correlation)
            row_wasserstein2.append(val_correlation)
            row_cramers_v2.append(val_correlation)
            row_total_variation2.append(val_correlation)
            row_js2.append(val_correlation)
            row_Casuality2.append(val_correlation)
            row_Combined2.append(val_correlation)
    data_writer.writerow(row_hellinger)
    data_writer.writerow(row_wasserstein)
    data_writer.writerow(row_cramers_v)
    data_writer.writerow(row_total_variation)
    data_writer.writerow(row_js)
    data_writer.writerow(row_Casuality)
    data_writer.writerow(row_Combined)

    data_writer2.writerow(row_hellinger2)
    data_writer2.writerow(row_wasserstein2)
    data_writer2.writerow(row_cramers_v2)
    data_writer2.writerow(row_total_variation2)
    data_writer2.writerow(row_js2)
    data_writer2.writerow(row_Casuality2)
    data_writer2.writerow(row_Combined2)

data_file.close()
data_file2.close()

