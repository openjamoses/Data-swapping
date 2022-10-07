import csv

import pandas as pd
import numpy as np
import operator
import math
def sort_dict(dict_, reverse=True):
    return dict(sorted(dict_.items(), key=operator.itemgetter(1), reverse=reverse))
class Ranking:
    @staticmethod
    def rank(data, data2):
        print(data.columns)
        Swap = data['Swap \n Percentage'].values
        Measurement = data.Measurement.values
        Feature = data.Feature.values
        SUM_COMBINED = data.SUM_COMBINED.values
        CDI = data.CDI.values

        Test_Percent = data2['Test \n Percentage'].values
        Feature2 = data2.Feature.values
        Shap_Value = data2.Value.values

        data_file2 = open(path_output2 + 'Stability-shap-{}.csv'.format(data_name),
                         mode='w', newline='',
                         encoding='utf-8')
        data_writer2 = csv.writer(data_file2)
        data_writer2.writerow(['Measures','Rank_CDI_SHAP', 'Rank_Double_SHAP'])
        set_Measurement = ['Hellinger\n distance', 'Jensen-Shannon\n divergence', 'Total variation\n distance',
                           'Wasserstein\n distance']  # set(Measurement)

        rank_data_shap = {}
        for i in range(len(Feature2)):
            #if Test_Percent[i] == '50%':
            if not Feature2[i] in rank_data_shap.keys():
                rank_data_shap[Feature2[i]] = {}
            rank_data_shap[Feature2[i]][Test_Percent[i]] = Shap_Value[i]


        rank_data_shap2 = {}
        for key, val in rank_data_shap.items():
            rank_data_shap2[key] = {}
            prev_shap = -1
            rank_index = 0
            sorted_data_shap = sort_dict(val, reverse=True)
            for key2, val2 in sorted_data_shap.items():
                if val2 != prev_shap:
                    rank_index += 1
                rank_data_shap2[key][key2] = rank_index

        rank_data = {}
        rank_data_ = {}
        for i in range(len(Feature)):
            if not Measurement[i] in rank_data.keys():
                rank_data[Measurement[i]] = {}
                rank_data_[Measurement[i]] = {}
            if not Feature[i] in rank_data[Measurement[i]].keys():
                rank_data[Measurement[i]][Feature[i]] = {}
                rank_data_[Measurement[i]][Feature[i]] = {}
            rank_data[Measurement[i]][Feature[i]][Swap[i]] = SUM_COMBINED[i]
            rank_data_[Measurement[i]][Feature[i]][Swap[i]] = CDI[i]
        #print(Swap)
        #for measure_2 in set_Measurement:
        rank_data2_ = {}
        for key, val in rank_data_.items():
            rank_data2_[key] = {}
            for key_, val_ in val.items():
                sorted_data = sort_dict(val_, reverse=True)
                #print(key, sorted_data)
                rank_data2_[key][key_] = {}
                index = 0
                prev_val = -1
                for key2, val2 in sorted_data.items():
                    if val2 != prev_val:
                        prev_val = val2
                        index += 1
                    rank_data2_[key][key_][key2] = index

        rank_data2 = {}

        for key, val in rank_data.items():
            rank_data2[key] = {}
            for key_, val_ in val.items():
                sorted_data = sort_dict(val_, reverse=True)
                #print(key, sorted_data)
                rank_data2[key][key_] = {}
                index = 0
                prev_val = -1
                for key2, val2 in sorted_data.items():
                    if val2 != prev_val:
                        prev_val = val2
                        index += 1
                    rank_data2[key][key_][key2] = index
        Feature_set = set(Feature)
        m = len(Feature_set)
        print(set_Measurement)

        for measure_1 in set_Measurement:
            #val = rank_data.get(measure_1)
            row = [measure_1]
            list_single = []
            list_double = []
            for key2, val2 in rank_data[measure_1].items():
                sum_rank = 0
                sum_rank_single = 0
                for key2_, val2_ in val2.items():
                    sum_rank += (math.pow((rank_data2[measure_1][key2][key2_] - rank_data_shap2[key2][key2_]), 2)) / (
                                m * (math.pow(m, 2) - 1))
                    sum_rank_single += (math.pow((rank_data2_[measure_1][key2][key2_] - rank_data_shap2[key2][key2_]), 2)) / (
                            m * (math.pow(m, 2) - 1))
                list_single.append(1-sum_rank_single)
                list_double.append(1-sum_rank)
            row.append('{} ({})'.format(round(np.mean(list_single),3), round(np.std(list_single),3)))
            row.append('{} ({})'.format(round(np.mean(list_double),3), round(np.std(list_double),3)))
            data_writer2.writerow(row)


if __name__ == '__main__':
    path = '../../../dataset/'
    alpha = 0.3
    #clevelan_heart
    data_name = 'Student-{}_'.format(alpha)  # _35_threshold
    path_output = 'logging3/{}/'.format(data_name)
    path_output2 = '/Volumes/Cisco/Fall2022/Fairness/Analysis/Ranking/Shap/'
    data = pd.read_csv(path+path_output +'/processed'+'/Correlated_features_{}.csv'.format(data_name))
    data2 = pd.read_csv(path + path_output + '/processed' + '/Shap_importance_{}.csv'.format(data_name))
    ranking = Ranking.rank(data, data2)