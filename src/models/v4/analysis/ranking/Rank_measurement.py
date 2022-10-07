import csv

import pandas as pd
import numpy as np
import operator
import math
def sort_dict(dict_, reverse=True):
    return dict(sorted(dict_.items(), key=operator.itemgetter(1), reverse=reverse))
class Ranking:
    @staticmethod
    def rank(data):
        print(data.columns)
        Swap = data['Swap \n Percentage'].values
        Measurement = data.Measurement.values
        Feature = data.Feature.values
        SUM_COMBINED = data.SUM_COMBINED.values

        data_file = open(path_output2+'Rank_{}.csv'.format(data_name),
            mode='w', newline='',
            encoding='utf-8')
        data_writer = csv.writer(data_file)

        data_file2 = open(path_output2 + 'Stability_{}.csv'.format(data_name),
                         mode='w', newline='',
                         encoding='utf-8')
        data_writer2 = csv.writer(data_file2)
        set_Measurement = ['Hellinger\n distance','Jensen-Shannon\n divergence', 'Total variation\n distance',  'Wasserstein\n distance'] #set(Measurement)
        row = ['Feature']
        for measure in set_Measurement:
            row.append(measure)
        data_writer.writerow(row)

        row = ['Measures']
        for measure in set_Measurement:
            row.append(measure)
        data_writer2.writerow(row)

        rank_data = {}
        for i in range(len(Feature)):
            if Swap[i] == '50%':
                if Measurement[i] in rank_data.keys():
                    rank_data[Measurement[i]][Feature[i]] = SUM_COMBINED[i]
                else:
                    rank_data[Measurement[i]] = {}
                    rank_data[Measurement[i]][Feature[i]] = SUM_COMBINED[i]
        #print(Swap)

        #for measure_2 in set_Measurement:


        rank_data2 = {}

        for key, val in rank_data.items():
            sorted_data = sort_dict(val, reverse=True)
            #print(key, sorted_data)
            rank_data2[key] = {}
            index = 1
            for key2, val2 in sorted_data.items():
                rank_data2[key][key2] = index
                index += 1

        for feat in set(Feature):
            row = [feat]
            for measure_ in set_Measurement:
                val_ = rank_data2[measure_][feat]
                row.append(val_)
            data_writer.writerow(row)

        m = len(set(Feature))
        print(set_Measurement)
        for measure_1 in set_Measurement:
            #val = rank_data.get(measure_1)
            row = [measure_1]
            val_ = []
            for measure_2 in set_Measurement:
                if measure_1 != measure_2:
                    sum_rank = 0
                    for key2, val2 in rank_data[measure_1].items():
                        sum_rank += (math.pow((rank_data2[measure_1][key2] - rank_data2[measure_2][key2]),2))/(m*(math.pow(m,2)-1))
                    #v = round((1-sum_rank))
                    row.append(1-sum_rank)
                    val_.append(1-sum_rank)
                else:
                    row.append('-')
            row.append(np.mean(val_))
            data_writer2.writerow(row)


if __name__ == '__main__':
    path = '../../../dataset/'
    alpha = 0.3
    data_name = 'compas-{}_'.format(alpha)  # _35_threshold
    path_output = 'logging3/{}/'.format(data_name)
    path_output2 = '/Volumes/Cisco/Fall2022/Fairness/Analysis/Ranking/'
    data = pd.read_csv(path+path_output +'/processed'+'/Correlated_features_{}.csv'.format(data_name))

    ranking = Ranking.rank(data)