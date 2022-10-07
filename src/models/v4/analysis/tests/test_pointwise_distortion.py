from itertools import combinations
import math
import numpy as np
#from src.models.v3.load_transformed_data import LoadData
from scipy import stats
import random
from src.models.v3.load_data import LoadData
from src.models.v3.utility_functions import data_split, split_features_target
from src.models.v4.distance_measure2 import DistanceMeasure2

def normal_pdf(x):
    return stats.norm.pdf(x) #math.exp(-(x-mu) ** 2 / 2 / sigma ** 2) / (SQRT_TWO_PI * sigma))

class Distance:

    @staticmethod
    def gowa_distance(p, q, discrete_indices=[]):
        distance_list = []
        list_index = []
        for i in range(len(p)):
            distance = 0
            if i in discrete_indices:
                if p[i] != q[i]:
                    distance = 1
                distance_list.append(distance)
            else:
                list_index.append(i)

        data_p = {}
        data_q = {}
        for i,j in combinations(list_index, 2):
            if i in data_p.keys():
                data_p[i].append([p[i], p[j]])
                data_q[i].append([q[i], q[j]])
            else:
                data_p[i] = [[p[i], p[j]]]
                data_q[i] = [[q[i], q[j]]]
        for index, data_ in data_p.items():
            list_p_i = []
            list_p_j = []

            list_q_i = []
            list_q_j = []
            euc_p = 0
            euc_q = 0
            for val in data_:
                euc_p += (val[0]-val[1])**2
            for val in data_q.get(index):
                euc_q += (val[0]-val[1])**2
            pf = 0
            if euc_q != euc_p:
                if euc_p != 0:
                    #if euc_q == 0:
                    #    euc_q = 0.1
                    #print(euc_q, euc_p)
                    pf = math.sqrt(euc_q)/ math.sqrt(euc_p)
                pf = abs(pf-1)


            distance_list.append(pf)
        #print(distance_list)
        return np.mean(distance_list)

    @staticmethod
    def sigma_(p, q, discrete_indices=[]):
        distance_sum = 0
        p = normal_pdf(p)
        q = normal_pdf(q)
        n = len(p)
        #list_val = []
        alpha = 1
        discrete_sum = 0
        list_cont_indices = []
        list_ = []
        for i in range(len(p)):
            if i in discrete_indices:
                if p[i] != q[i]:
                    discrete_sum += 1
            else:
                list_cont_indices.append(i)
        data_p = {}
        data_q = {}
        list_p_1 = []
        list_p_2 = []

        list_q_1 = []
        list_q_2 = []
        for i,j in combinations(list_cont_indices, 2):
            if i in data_p.keys():
                data_p[i].append([p[i], p[j]])
                data_q[i].append([q[i], q[j]])
            else:
                data_p[i] = [[p[i], p[j]]]
                data_q[i] = [[q[i], q[j]]]
            list_p_1.append(p[i])
            list_p_2.append(p[j])

            list_q_1.append(q[i])
            list_q_2.append(q[j])
        dist_ = (DistanceMeasure2.total_variation_distance(list_p_1, list_p_2))/(DistanceMeasure2.total_variation_distance(list_q_1, list_q_2))

        pf_sum = 0
        for index, data_ in data_p.items():
            euc_p = 0
            euc_q = 0
            for val in data_:
                euc_p += (val[0]-val[1])**2
            for val in data_q.get(index):
                euc_q += (val[0]-val[1])**2
            #print(' ---index: ', index, data_, euc_p, euc_q)
            if euc_q != 0:
                pf = math.sqrt(euc_p)/math.sqrt(euc_q)
                pf_sum += pf
        for index, data_ in data_p.items():
            euc_p = 0
            euc_q = 0
            for val in data_:
                euc_p += (val[0] - val[1]) ** 2
            for val in data_q.get(index):
                euc_q += (val[0] - val[1]) ** 2
            if euc_q != 0:
                pf = math.sqrt(euc_p) / math.sqrt(euc_q)
            else:
                pf = 0

            #pf = math.sqrt(euc_p) / math.sqrt(euc_q)
            #print(p,q)
            pf_ = (n * (n-1)*pf)/pf_sum

            #print('    ----: ',index, pf, pf_)

            #pf = (p[i]-q[i])**2
            #if pf > 1:
            list_.append(pf_)
        #for i in range(len(p)):
        '''min_val = 1
        if len(list_)>0:
            min_val = min(list_)

        for i in range(len(p)):
            pf = ((p[i] - q[i])**2)/ min_val
            distance_sum += pf
        normalize_avg = (2 * distance_sum)/(n*(n-1))'''
        print('total variation: ', dist_, p, q)
        return distance_sum/len(p), np.var(list_)
    @staticmethod
    def sigma_distortion(p,q):
        print("\n\n*********\n")
        list_distances = []
        n = len(p)
        dict_x = {}
        dict_y = {}
        #p_ = normal_pdf(p)
        #q_ = normal_pdf(q)
        x = p
        y = q

        data = {}
        data_y = {}
        for i, j in combinations(range(len(x)), 2):
            if i in data.keys():
                data[i].append([x[i], x[j]])
                data_y[i].append([y[i], y[j]])
            else:
                data[i] = [[x[i], x[j]]]
                data_y[i] = [[y[i] , y[j]]]
            euc_x_temp = pow((x[i] - x[j]), 2)
            euc_y_temp = pow((y[i] - y[j]), 2)
            if euc_x_temp != 0:
                pf = euc_y_temp/euc_x_temp
            else:
                pf = 0
            print(' ---- pf: ',x[i], x[j], y[i], y[j], pf)

        pf_sum = 0
        for index, val in data.items():
            euc_x = 0
            euc_y = 0
            for val_ in val:
                #print(val_)
                euc_x += pow((val_[0] - val_[1]), 2)
            for val2_ in data_y[index]:
                euc_y += pow((val2_[0] - val2_[1]), 2)
            if euc_x != 0:
                pf = math.sqrt(euc_y) / math.sqrt(euc_x)
            else:
                pf = 0
            pf_sum += pf


        for index, val in data.items():
            euc_x = 0
            euc_y = 0
            for val_ in val:
                #print(val_)
                euc_x += pow((val_[0] - val_[1]), 2)
            for val2_ in data_y[index]:
                euc_y += pow((val2_[0] - val2_[1]), 2)
            if euc_x != 0:
                #pf = math.sqrt(euc_y) / math.sqrt(euc_x)
                pf = euc_y / euc_x
                pf_ = (n * (n - 1) * pf) / (2 * pf_sum)  # n*(n-1)*
            else:
                pf = 0
                pf_ = 0
            #print('     ---: ', index, pf, pf_)

            list_distances.append((pf-1)**2)
        #list_distances.append((pf_ - 1) ** 2)
        #print(pf_, pf, (pf_ - 1) ** 2)
        print(p, q)
        return np.mean(list_distances), np.var(list_distances)


if __name__ == '__main__':
    path = '../../../dataset/'
    correlation_threshold = 0.45
    loadData = LoadData(path, threshold=correlation_threshold)  # ,threshold=correlation_threshold
    data_name = 'adult_'  # _35_threshold

    data = loadData.load_adult_data('adult.data.csv')
    target_index = loadData.target_index

    train, test = data_split(data=data.to_numpy(), sample_size=0.25)
    x_train, y_train = split_features_target(train, index=target_index)
    x_test, y_test = split_features_target(test, index=target_index)

    column_names = data.columns.tolist()

    sensitive = ['sex', 'race']
    sensitive_indices = [column_names.index(sensitive[0]), column_names.index(sensitive[1])]
    cont_indices = [i for i in range(len(column_names)) if not i in sensitive_indices]
    column_index = 0
    x_test_altered = []

    print(' sensitive: ', sensitive_indices, ' continous: ', cont_indices)

    age_ = data.age.values.tolist()
    capital_gain = data['capital-gain'].values.tolist()
    capital_index = column_names.index('capital-gain')
    for i in range(len(x_test)):
        val = []
        for j in range(len(x_test[i])):
            val_new = x_test[i][j]
            if j == column_index:
                if x_test[i][j] == 0:
                    val_new = 1
                else:
                    val_new = 20 #random.randint(min(age_), max(age_))
            elif j == 1:
                if x_test[i][j] == 0:
                    val_new = 1
                else:
                    val_new = 0
               #elif j == len(x_test[i])-1:
               # val_new = 10 #random.randint(min(capital_gain), max(capital_gain))
            val.append(val_new)
        x_test_altered.append(val)

        random.randint(0, 9)


    #data = loadData.load_compas_data('compas-scores-two-years.csv')
    age = data['sex'].values.tolist()
    age_transformed = []
    for i in range(len(age)):
        if age[i] == 1:
            age_transformed.append(0)
        else:
            age_transformed.append(1)
    #for i in range(len(x_test)):

    for i in range(len(x_test)):
        distortion = Distance.gowa_distance(x_test[i], x_test_altered[i], sensitive_indices)
        hellinger = DistanceMeasure2.hellinger_continous_1d(x_test[i], x_test_altered[i])
        total_variation = DistanceMeasure2.total_variation_distance(x_test[i], x_test_altered[i])
        wasserstein = DistanceMeasure2.wasserstein_distance(x_test[i], x_test_altered[i])
        js_divergence = DistanceMeasure2.js_divergence(x_test[i], x_test_altered[i])
        #js_divergence = DistanceMeasure2.js_divergence(x_test[i], x_test_altered[i])
        #print('distortion: ', i, distortion, hellinger, total_variation, wasserstein, js_divergence)

        print(distortion, x_test[i], x_test_altered[i])