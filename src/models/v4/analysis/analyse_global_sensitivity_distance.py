import csv

import pandas as pd
import numpy as np


#todo: group by tutorials: https://stackoverflow.com/questions/17679089/pandas-dataframe-groupby-two-columns-and-get-counts

path = '../../dataset/'
#if __name__ == '__main__':
#df = pd.read_csv(path+'logging/log_KL_Divergence_overral_adult-45.csv')
df_global = pd.read_csv(path + 'logging3/KL_Divergence_global_adult.csv')

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

#todo: global
group_data = df_global.groupby(['ID', 'Feature','swap_proportion'])['Distortion_hellinger', 'Distortion_wasserstein', 'cramers_v_distortion',
'Distortion_js', 'hellinger_div', 'wasserstein_div','cramers_v_div','JS_Div', 'Casuality', 'Importance'].mean().reset_index()
#print(len(group_data))
#print(group_data.head())
#group_data.to_csv(path + 'logging3/processed/KL_Divergence_local_adult_processed_global.csv', index=False)
data_file = open(path + 'logging3/processed/KL_Divergence_local_adult_processed_global2_.csv', mode='w', newline='',
                 encoding='utf-8')
data_writer = csv.writer(data_file)
data_writer.writerow(['ID','Features','Swap', 'Swap_ratio', 'Distance_measure', 'Value'])
ID = group_data.ID.values.tolist()
Feature = group_data.Feature.values.tolist()
#Category = group_data.Category.values.tolist()
swap_proportion = group_data.swap_proportion.values.tolist()
Distortion_hellinger = group_data.Distortion_hellinger.values.tolist()
Distortion_wasserstein = group_data.Distortion_wasserstein.values.tolist()
cramers_v_distortion = group_data.cramers_v_distortion.values.tolist()
Distortion_js = group_data.Distortion_js.values.tolist()

hellinger_div = group_data.hellinger_div.values.tolist()
wasserstein_div = group_data.wasserstein_div.values.tolist()
cramers_v_div = group_data.cramers_v_div.values.tolist()
JS_Div = group_data.JS_Div.values.tolist()
for i in range(len(Feature)):
    data_writer.writerow([ID[i], Feature[i], swap_proportion[i], 'swat_{}'.format(swap_proportion[i]), 'Hellinger', hellinger_div[i]])
    data_writer.writerow(
        [ID[i], Feature[i],swap_proportion[i], 'swat_{}'.format(swap_proportion[i]), 'wasserstein',
         wasserstein_div[i]])
    data_writer.writerow(
        [ID[i], Feature[i], swap_proportion[i], 'swat_{}'.format(swap_proportion[i]), 'cramers_v',
         cramers_v_div[i]])

    data_writer.writerow(
        [ID[i], Feature[i],swap_proportion[i], 'swat_{}'.format(swap_proportion[i]), 'JS_Divergence',
         JS_Div[i]])
data_file.close()