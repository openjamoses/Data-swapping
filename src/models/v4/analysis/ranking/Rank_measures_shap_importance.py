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

        # feature_set = set(['race', 'sex', 'age', 'hours-per-week', 'capital-gain', 'capital-loss'])
        # feature_list = ['race', 'sex', 'age', 'hours-per-week', 'capital-gain', 'capital-loss'] #list(feature_set)
        feature_list = ['sex', 'age', 'thalach', 'ca', 'thal', 'exang', 'cp', 'trestbps', 'restecg', 'fbs', 'oldpeak',
                        'chol']  # list(feature_set)
        # feature_list = ['race', 'sex', 'age', 'hours-per-week', 'capital-gain', 'capital-loss'] #list(feature_set)
        #feature_list = ['sex','age','health','Pstatus','nursery','Medu', 'Fjob', 'schoolsup', 'absences',  'activities', 'higher', 'traveltime',  'paid', 'guardian',  'Walc', 'freetime', 'famsup',  'romantic', 'studytime', 'goout', 'reason',  'famrel', 'internet']
        # feature_list = [ 'Sex', 'Age','Job', 'Saving', 'Checking', 'Credit','Housing', 'Purpose']
        feature_list = ['race','sex', 'age', 'c_charge_degree', 'priors_count']
        #feature_list = ['age', 'education', 'job', 'loan', 'balance', 'housing', 'duration', 'campaign', 'default']


        Swap = data['Swap \n Percentage'].values
        Measurement = data.Measurement.values
        Feature = data.Feature.values
        SUM_COMBINED = data.SUM_COMBINED.values
        CDI = data.CDI.values

        Test_Percent = data2['Test \n Percentage'].values
        Feature2 = data2.Feature.values
        Shap_Value = data2.Value.values

        data_file = open(path_output2 + 'Ranking-value-{}.csv'.format(data_name),
                          mode='w', newline='',
                          encoding='utf-8')
        data_writer = csv.writer(data_file)
        data_writer.writerow(['Distance_Measure','Feature', 'Distance_Rank', 'Shap_Rant', 'Difference'])

        data_file2 = open(path_output2 + 'Stability-shap-{}.csv'.format(data_name),
                         mode='w', newline='',
                         encoding='utf-8')
        data_writer2 = csv.writer(data_file2)
        data_writer2.writerow(['Measures','Rank_CDI_SHAP', 'Rank_Double_SHAP'])
        set_Measurement = ['Hellinger\n distance', 'Jensen-Shannon\n divergence', 'Total variation\n distance',
                           'Wasserstein\n distance']  # set(Measurement)

        rank_data_shap = {}
        for i in range(len(Feature2)):
            if Test_Percent[i] == '50%':
                rank_data_shap[Feature2[i]] = Shap_Value[i]
        prev_shap = -1
        rank_index = 0
        rank_data_shap2 = {}
        sorted_data_shap = sort_dict(rank_data_shap, reverse=True)
        for key, val in sorted_data_shap.items():
            if val != prev_shap:
                rank_index += 1
            rank_data_shap2[key] = rank_index

        rank_data = {}
        rank_data_ = {}
        for i in range(len(Feature)):
            if Swap[i] == '50%':
                if Measurement[i] in rank_data.keys():
                    rank_data[Measurement[i]][Feature[i]] = SUM_COMBINED[i]
                    rank_data_[Measurement[i]][Feature[i]] = CDI[i]
                else:
                    rank_data[Measurement[i]] = {}
                    rank_data[Measurement[i]][Feature[i]] = SUM_COMBINED[i]

                    rank_data_[Measurement[i]] = {}
                    rank_data_[Measurement[i]][Feature[i]] = CDI[i]
        #print(Swap)
        #for measure_2 in set_Measurement:
        rank_data2_ = {}
        for key, val in rank_data_.items():
            sorted_data = sort_dict(val, reverse=True)
            #print(key, sorted_data)
            rank_data2_[key] = {}
            index = 0
            prev_val = -1
            for key2, val2 in sorted_data.items():
                if val2 != prev_val:
                    prev_val = val2
                    index += 1
                rank_data2_[key][key2] = index

        rank_data2 = {}

        for key, val in rank_data.items():
            sorted_data = sort_dict(val, reverse=True)
            #print(key, sorted_data)
            rank_data2[key] = {}
            index = 0
            prev_val = -1
            for key2, val2 in sorted_data.items():
                if val2 != prev_val:
                    prev_val = val2
                    index += 1
                rank_data2[key][key2] = index
        Feature_set = set(Feature)
        m = len(Feature_set)
        print(set_Measurement)

        for measure_1 in set_Measurement:
            #val = rank_data.get(measure_1)
            row = [measure_1]
            val_ = []
            sum_rank = 0
            sum_rank_single = 0
            for key2 in feature_list:
                val2 = rank_data[measure_1][key2]
                #for key2, val2 in rank_data[measure_1].items():
                sum_rank += (math.pow((rank_data2[measure_1][key2] - rank_data_shap2[key2]), 2)) / (
                            m * (math.pow(m, 2) - 1))
                sum_rank_single += (math.pow((rank_data2_[measure_1][key2] - rank_data_shap2[key2]), 2)) / (
                        m * (math.pow(m, 2) - 1))
                data_writer.writerow([measure_1, key2, rank_data2[measure_1][key2], rank_data_shap2[key2], abs((rank_data2[measure_1][key2] - rank_data_shap2[key2]))])
            row.append(1 - round(sum_rank_single,3))
            row.append(1 - round(sum_rank,3))
            data_writer2.writerow(row)


if __name__ == '__main__':
    path = '../../../dataset/'
    alpha = 0.3
    #clevelan_heart
    data_name = 'compas-{}_'.format(alpha)  # _35_threshold
    path_output = 'logging3/{}/'.format(data_name)
    path_output2 = '/Volumes/Cisco/Fall2022/Fairness/Analysis/Ranking/Shap/'
    data = pd.read_csv(path+path_output +'/processed'+'/Correlated_features_{}.csv'.format(data_name))
    data2 = pd.read_csv(path + path_output + '/processed' + '/Shap_importance_{}.csv'.format(data_name))
    ranking = Ranking.rank(data, data2)