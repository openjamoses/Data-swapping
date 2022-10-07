import csv
import os
import random

import xgboost
from sklearn import model_selection
from tensorflow.python.keras.utils.np_utils import to_categorical

from src.models.data.kl_jn_divergences import kl_js_divergence_metrics
from src.models.data.load_train_test import LoadTrainTest
from src.models.data.models import Models, Models2
from src.models.v3.NNClassifier import NNClassifier
from src.models.v3.load_data import LoadData
from src.models.v3.ranking import Ranking
from src.models.v4.distance_measure2 import DistanceMeasure2
from src.models.v4.sensitivity_utils_4_3 import *
from src.models.v3.utility_functions import *
import seaborn as sn
import matplotlib.pyplot as plt
import tensorflow as tf

from src.models.v4.distance_measure import DistanceMeasure


class SensitivityAnalysis:
    def __init__(self, data,df_data, target_index=None, sensitive_name='sex', sensitive_index=8, log_data=True,
                 log_path='../dataset/logging3/', data_name='', colums_list=[], threshold=None):
        self.data = data
        self.df_data = df_data
        self.colums_list = colums_list
        self.sensitive_name = sensitive_name
        self.sensitive_index = sensitive_index
        self.target_index = target_index
        self.posible_sensitive_values = list(np.unique(self.data[0:, self.sensitive_index]))
        self.data_init(data)

        self.log_data = log_data
        self.new_path = log_path
        #if self.log_data:
        if threshold == None:
            id = '-baseline'
        else:
            id = threshold
        if not os.path.exists(log_path + str(data_name)):
            os.makedirs(log_path + str(data_name))
        if not os.path.exists(log_path + data_name + "/{}".format(id)):
            os.makedirs(log_path + str(data_name) + "/{}".format(id))
        self.new_path = log_path + str(data_name) + "/{}/".format(id)
        self.log_path_global = log_path + 'KL_Divergence_global_{}.csv'.format(data_name)
        self.data_file = open(self.log_path_global, mode='w', newline='',
                              encoding='utf-8')
        self.data_writer = csv.writer(self.data_file)
        self.data_writer.writerow(
            ['ID', 'Feature','Category', 'swap_proportion', 'Acc', 'Acc_after','distortion_multiple','distortion_feature', 'Distortion_hellinger','Distortion_wasserstein','cramers_v_distortion', 'total_variation_distortion',  'Distortion_kl', 'Distortion_js', 'hellinger_div','wasserstein_div','cramers_v_div', 'total_variation_div', 'KL_div', 'JS_Div', 'Casuality', 'Importance', 'SP', 'SP_after',
             'TPR', 'TPR_after', 'FPR', 'FPR_after'])

        self.log_path_local = log_path + 'KL_Divergence_local_{}.csv'.format(data_name)
        self.data_file2 = open(self.log_path_local, mode='w', newline='',
                               encoding='utf-8')
        self.data_writer2 = csv.writer(self.data_file2)
        #self.data_writer2.writerow(
        #    ['ID', 'Feature', 'Category', 'Acc', 'Acc_after', 'Distortion_hellinger','Distortion_wasserstein', 'Distortion_kl', 'Distortion_js', 'KL_Pos', 'JS_Divergence', 'Casuality', 'Importance', 'SP',
        #     'SP_after', 'TPR', 'TPR_after', 'FPR', 'FPR_after'])
        self.data_writer2.writerow(
            ['ID', 'Feature', 'Category','swap_proportion','Acc', 'Acc_after', 'distortion_multiple','distortion_feature', 'Distortion_hellinger','Distortion_wasserstein','cramers_v_distortion', 'total_variation_distortion', 'Distortion_kl', 'Distortion_js', 'hellinger_div','wasserstein_div', 'cramers_v_div','total_variation_div', 'KL_div', 'JS_Div', 'Casuality', 'Importance',
             'SP','SP_after', 'TPR', 'TPR_after', 'FPR', 'FPR_after'])

        #data_category[category]['total_variation_distortion'] = total_variation_distortion
        #print('Sensitive attribute: ', sensitive_name)

    def data_init(self, data):
        self.train, self.test = data_split(data=data, sample_size=0.25)
        self.x_train, self.y_train = split_features_target(self.train, index=self.target_index)
        self.x_test, self.y_test = split_features_target(self.test, index=self.target_index)

        print('self.y_test: ', self.y_test)

    def get_categorical_features(self, feature_index):
        #print('column data: ', set(self.data[0:, feature_index]))
        return list(set(self.data[0:, feature_index]))
    def get_features_by_index(self, data_, feature_index):
        #print('column data: ', set(self.data[0:, feature_index]))
        feature_data = np.array(data_)[0:, feature_index]
        return [val for val in feature_data]

    def get_categorical_features_posible_value(self, feature_index):
        #print('column data: ', set(self.data[0:, feature_index]))
        data_folded = {}
        for val in list(set(self.data[0:, feature_index])):
            data_folded[val] = val
        return data_folded  # list(set(self.data[0:, feature_index]))

    def determine_range(self, feature_index, k_folds=3):
        data_range = self.data[0:, feature_index]
        # interval_ = round(((max(sorted_)-min(sorted_))/k_folds),0)
        fold_count = 0
        folded_data = {}
        # percentile_25 = np.percentile(data_range, 25)
        percentile_50 = np.percentile(data_range, 50)
        percentile_75 = np.percentile(data_range, 75)
        percentile_100 = np.percentile(data_range, 100)
        if percentile_50 == np.min(data_range) or percentile_50 == np.max(data_range):
            # percentile_25 = np.percentile(np.unique(data_range), 25)
            percentile_50 = np.percentile(np.unique(data_range), 50)
            # percentile_75 = np.percentile(np.unique(data_range), 75)
        # if percentile_50 == percentile_25:
        #    percentile_50 = np.max(data_range)/2
        # if percentile_25 == np.min(data_range):
        #    percentile_25 = percentile_50/2
        for i in range(len(data_range)):
            fold_id = percentile_50
            if data_range[i] <= percentile_50:
                fold_id = percentile_50
            elif data_range[i] > percentile_50:  # and data_range[i] <= percentile_50:
                fold_id = percentile_100
            # elif data_range[i] > percentile_50:
            #    fold_id = percentile_75
            if fold_id in folded_data.keys():
                folded_data[fold_id].add(data_range[i])
            else:
                folded_data[fold_id] = set([data_range[i]])
        for key, val in folded_data.items():
            if len(val) < 3:
                #print('Category <3: ', np.min(list(val)), percentile_50, percentile_100)
                if key == percentile_50:
                    val2 = []
                    val2.append(random.uniform(np.min(list(val)), np.max(list(val))))
                    val2.append(random.uniform(np.min(list(val)), np.max(list(val))))
                    val2.extend(list(val))
                if key == percentile_50:
                    val2 = []
                    # print('percentile_50: ', percentile_50, val)
                    # val2.append(random.uniform(percentile_50, np.min(list(val))))
                    # val2.append(random.uniform(percentile_50, np.min(list(val))))
                    val2.append(random.uniform(percentile_50, np.max(list(val))))
                    val2.append(random.uniform(percentile_50, np.max(list(val))))
                    val2.extend(list(val))
                '''if key == percentile_75:
                    val2 = []
                    val2.append(random.uniform(percentile_50, percentile_100))
                    val2.append(random.uniform(percentile_50, percentile_100))
                    val2.extend(list(val))'''
                folded_data[key] = list(val)
            else:
                folded_data[key] = list(val)
        return folded_data  # list(folded_data.values())
    def compute_metrics(self, y_predicted_before, y_predicted_after, column_id, p_pred,
                        q_pred, distortion_multiple, distortion_feature, hellinger_distortion, wasserstein_distance_distortion,cramers_v_distortion, total_variation_distortion, kl_div_distortion, js_div_distortion, runkl=False):

        p_pred = abs(p_pred)
        q_pred = abs(q_pred)
        # for column_id in range(self.x_train.shape[1]):
        #    if is_categorical(self.x_test,column_id):
        data_category = {}
        category_list = self.get_categorical_features(column_id)
        for category in category_list:
            category_target_original = []
            category_target_predicted_before = []
            category_target_predicted_after = []
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            TP_after = 0
            FP_after = 0
            TN_after = 0
            FN_after = 0
            a_before_sub, b_before_sub = [], []
            a_after_sub, b_after_sub = [], []

            for i in range(len(self.x_test)):
                if self.x_test[i][column_id] == category:
                    category_target_original.append(self.y_test[i])
                    category_target_predicted_before.append(y_predicted_before[i])
                    category_target_predicted_after.append(y_predicted_after[i])
                    if y_predicted_before[i] == 1 and self.y_test[i] == 1:
                        TP += 1
                    if y_predicted_before[i] == 1 and self.y_test[i] == 0:
                        FP += 1
                    if y_predicted_before[i] == 0 and self.y_test[i] == 0:
                        TN += 1
                    if y_predicted_before[i] == 0 and self.y_test[i] == 1:
                        FN += 1
                    if y_predicted_after[i] == 1 and self.y_test[i] == 1:
                        TP_after += 1
                    if y_predicted_after[i] == 1 and self.y_test[i] == 0:
                        FP_after += 1
                    if y_predicted_after[i] == 0 and self.y_test[i] == 0:
                        TN_after += 1
                    if y_predicted_after[i] == 0 and self.y_test[i] == 1:
                        FN_after += 1

                    #if runkl:
                    a_before_sub.append(p_pred[i][0])
                    b_before_sub.append(p_pred[i][1])
                    # if q_pred != None:
                    a_after_sub.append(q_pred[i][0])
                    b_after_sub.append(q_pred[i][1])
            category_target_original = np.array(category_target_original)
            category_target_predicted_before = np.array(category_target_predicted_before)
            category_target_predicted_after = np.array(category_target_predicted_after)
            acc_category = np.sum(category_target_original == category_target_predicted_before)
            TPR = calculate_TPR(TP, FP, TN, FN)
            FPR = calculate_FPR(TP, FP, TN, FN)
            SP = calculate_SP(TP, FP, TN, FN)

            acc_category_after = np.sum(category_target_original == category_target_predicted_after)
            f_importance = np.sum(category_target_original != category_target_predicted_after)
            TPR_after = calculate_TPR(TP_after, FP_after, TN_after, FN_after)
            FPR_after = calculate_FPR(TP_after, FP_after, TN_after, FN_after)
            SP_after = calculate_SP(TP_after, FP_after, TN_after, FN_after)

            impont = round(f_importance * 100 / len(category_target_original), 3)
            if str(impont) == 'nan' or len(category_target_original) == 0:
                print('nan found! ',category, category_list, column_id,  category_target_original)



            proportion = np.sum(
                category_target_predicted_before != category_target_predicted_after)  # calculate_proportion(TP, FP, TN, FN)

            kl_mean, js_mean, hellinger_mean, wasserstein_mean, cramers_v_mean, total_variation_mean = 0, 0, 0, 0, 0, 0
            if len(a_before_sub) > 1 and len(b_before_sub) > 0:
                # todo: KL Divergence
                kl_pos, js_pos = DistanceMeasure2.js_divergence(a_before_sub, a_after_sub)
                kl_neg, js_neg = DistanceMeasure2.js_divergence(b_before_sub, b_after_sub)

                # todo: Hellinger Distance
                hellinger_pos = DistanceMeasure.hellinger_continous_1d(a_before_sub, a_after_sub)
                hellinger_neg = DistanceMeasure.hellinger_continous_1d(b_before_sub, b_after_sub)
                # todo: wasserstein Distance
                wasserstein_distance_pos = DistanceMeasure2.wasserstein_distance(a_before_sub, a_after_sub)
                wasserstein_distance_neg = DistanceMeasure2.wasserstein_distance(b_before_sub, b_after_sub)

                # todo: cramers_v Distance
                #cramers_v_pos = DistanceMeasure.cramers_v(a_before_sub, a_after_sub)
                #cramers_v_neg = DistanceMeasure.cramers_v(b_before_sub, b_after_sub)

                # todo: total variation distance
                total_variation_distance_pos = DistanceMeasure2.total_variation_distance(a_before_sub, a_after_sub)
                total_variation_distance_neg = DistanceMeasure2.total_variation_distance(b_before_sub, b_after_sub)

                kl_mean = np.mean([kl_pos, kl_neg])
                js_mean = np.mean([js_pos, js_neg])
                hellinger_mean = np.mean([hellinger_pos, hellinger_neg])
                wasserstein_mean = np.mean([wasserstein_distance_pos, wasserstein_distance_neg])
                #cramers_v_mean = np.mean([cramers_v_pos, cramers_v_neg])
                total_variation_mean = np.mean([total_variation_distance_pos, total_variation_distance_neg])

            elif len(a_before_sub) > 1:
                kl_pos, js_pos = DistanceMeasure2.js_divergence(a_before_sub, a_after_sub)
                hellinger_pos = DistanceMeasure.hellinger_continous_1d(a_before_sub, a_after_sub)
                wasserstein_distance_pos = DistanceMeasure2.wasserstein_distance(a_before_sub, a_after_sub)
                #cramers_v_pos = DistanceMeasure.cramers_v(a_before_sub, a_after_sub)
                total_variation_distance_pos = DistanceMeasure2.total_variation_distance(a_before_sub, a_after_sub)

                kl_mean = kl_pos
                js_mean = js_pos
                hellinger_mean = hellinger_pos
                wasserstein_mean = wasserstein_distance_pos
                #cramers_v_mean = cramers_v_pos
                total_variation_mean = total_variation_distance_pos
            elif len(b_before_sub) > 1:
                kl_neg, js_neg = DistanceMeasure2.js_divergence(b_before_sub, b_after_sub)
                hellinger_neg = DistanceMeasure.hellinger_continous_1d(b_before_sub, b_after_sub)
                wasserstein_distance_neg = DistanceMeasure2.wasserstein_distance(b_before_sub, b_after_sub)
                #cramers_v_neg = DistanceMeasure.cramers_v(b_before_sub, b_after_sub)
                total_variation_distance_neg = DistanceMeasure2.total_variation_distance(b_before_sub, b_after_sub)

                kl_mean = kl_neg
                js_mean = js_neg
                hellinger_mean = hellinger_neg
                wasserstein_mean = wasserstein_distance_neg
                #cramers_v_mean = cramers_v_neg
                total_variation_mean = total_variation_distance_neg
            cramers_v_mean = DistanceMeasure.cramers_v(category_target_predicted_before,
                                                       category_target_predicted_after)
            if len(category_target_original)>0:
                #print('total_variation_mean: ', total_variation_mean, hellinger_mean, a_before_sub, b_before_sub)
                #print('cramers_v target: ', cramers_v_pos, cramers_v_neg)
                data_category[category] = {}
                data_category[category]['ACC'] = round(acc_category * 100 / len(category_target_original), 3)
                data_category[category]['TPR'] = TPR
                data_category[category]['FPR'] = FPR
                #hellinger_distortion, wasserstein_distance_distortion,kl_div_distortion, js_div_distortion
                data_category[category]['distortion_multiple'] = distortion_multiple
                data_category[category]['distortion_feature'] = distortion_feature
                data_category[category]['hellinger_distortion'] = hellinger_distortion
                data_category[category]['wasserstein_distance_distortion'] = wasserstein_distance_distortion
                data_category[category]['cramers_v_distortion'] = cramers_v_distortion
                data_category[category]['total_variation_distortion'] = total_variation_distortion
                data_category[category]['kl_div_distortion'] = kl_div_distortion
                data_category[category]['js_div_distortion'] = js_div_distortion

                data_category[category]['SP'] = SP
                data_category[category]['ACC_after'] = round(acc_category_after * 100 / len(category_target_original), 3)
                data_category[category]['TPR_after'] = TPR_after
                data_category[category]['FPR_after'] = FPR_after
                data_category[category]['SP_after'] = SP_after
                data_category[category]['proportion'] = round(proportion * 100 / len(category_target_original), 3)
                data_category[category]['Importance'] = round(f_importance * 100 / len(category_target_original), 3)
                data_category[category]['hellinger_div'] = hellinger_mean
                data_category[category]['wasserstein_distance_div'] = wasserstein_mean
                data_category[category]['cramers_v_div'] = cramers_v_mean
                data_category[category]['total_variation_div'] = total_variation_mean

                data_category[category]['KL_Divergence'] = kl_mean
                data_category[category]['JS_Divergence'] = js_mean

            #print('NAN category_target_original: ', acc_category_after, round(f_importance * 100 / len(category_target_original), 3), category_target_original,
            #      category_target_predicted_after)
        return data_category

    def compute_metrics_continous(self, y_predicted_before, y_predicted_after, column_id, p_pred,
                                  q_pred, distortion_multiple, distortion_feature, hellinger_distortion, wasserstein_distance_distortion,cramers_v_distortion,total_variation_distortion, kl_div_distortion, js_div_distortion, runkl=False):
        # for column_id in range(self.x_train.shape[1]):
        #    if is_categorical(self.x_test,column_id):
        p_pred = abs(p_pred)
        q_pred = abs(q_pred)
        data_category = {}
        category_data = self.determine_range(column_id)

        list_val = []
        for key, val in category_data.items():
            list_val.extend(val)
        #va_list = [vv for vv in list(category_data.values())]
        min_val = np.min(list_val)

        #print('category_data: ', category_data)
        for key, val in category_data.items():
            category_target_original = []
            category_target_predicted_before = []
            category_target_predicted_after = []
            TP = 0
            FP = 0
            TN = 0
            FN = 0

            TP_after = 0
            FP_after = 0
            TN_after = 0
            FN_after = 0
            a_before_sub, b_before_sub = [], []
            a_after_sub, b_after_sub = [], []
            for i in range(len(self.x_test)):
                if self.x_test[i][column_id] in val:
                    category_target_original.append(self.y_test[i])
                    category_target_predicted_before.append(y_predicted_before[i])
                    category_target_predicted_after.append(y_predicted_after[i])
                    # print(y_predicted[i], len(self.y_test[i]), self.y_test[i], self.y_test)
                    if y_predicted_before[i] == 1 and self.y_test[i] == 1:
                        TP += 1
                    if y_predicted_before[i] == 1 and self.y_test[i] == 0:
                        FP += 1
                    if y_predicted_before[i] == 0 and self.y_test[i] == 0:
                        TN += 1
                    if y_predicted_before[i] == 0 and self.y_test[i] == 1:
                        FN += 1
                    if y_predicted_after[i] == 1 and self.y_test[i] == 1:
                        TP_after += 1
                    if y_predicted_after[i] == 1 and self.y_test[i] == 0:
                        FP_after += 1
                    if y_predicted_after[i] == 0 and self.y_test[i] == 0:
                        TN_after += 1
                    if y_predicted_after[i] == 0 and self.y_test[i] == 1:
                        FN_after += 1
                    #if runkl:
                    a_before_sub.append(p_pred[i][0])
                    b_before_sub.append(p_pred[i][1])
                    # if q_pred != None:
                    a_after_sub.append(q_pred[i][0])
                    b_after_sub.append(q_pred[i][1])
            category_target_original = np.array(category_target_original)
            category_target_predicted_before = np.array(category_target_predicted_before)
            category_target_predicted_after = np.array(category_target_predicted_after)
            acc_category = np.sum(category_target_original == category_target_predicted_before)
            TPR = calculate_TPR(TP, FP, TN, FN)
            FPR = calculate_FPR(TP, FP, TN, FN)
            SP = calculate_SP(TP, FP, TN, FN)

            acc_category_after = np.sum(category_target_original == category_target_predicted_after)
            #if str(acc_category_after) == 'nan':
            f_importance = np.sum(category_target_original != category_target_predicted_after)
            TPR_after = calculate_TPR(TP_after, FP_after, TN_after, FN_after)
            FPR_after = calculate_FPR(TP_after, FP_after, TN_after, FN_after)
            SP_after = calculate_SP(TP_after, FP_after, TN_after, FN_after)

            #if len(category_target_original) == 0:
            #    print('f_importance: ', category, f_importance, category_target_original, category_target_predicted_after)

            proportion = np.sum(
                category_target_predicted_before != category_target_predicted_after)  # calculate_proportion(TP, FP, TN, FN)
            kl_pos = None
            kl_neg = None
            if len(val)>0 and len(category_target_predicted_before)> 0:
                if len(val) == 1:
                    category = str(val[0])
                else:
                    #max_val = np.max(list(category_data.values()))
                    if np.min(val) == min_val:
                        category = '<=' + str(np.max(val))
                    else:
                        category = '>=' + str(np.min(val))
                    #category = str(np.min(val)) + '-' + str(np.max(val))
                #if runkl
                #print('before call: ', a_before_sub, a_after_sub)
                kl_mean, js_mean, hellinger_mean, wasserstein_mean, cramers_v_mean, total_variation_mean = 0, 0, 0, 0, 0, 0


                if len(a_before_sub) >1 and len(b_before_sub) > 0:
                    # todo: KL Divergence
                    kl_pos, js_pos = DistanceMeasure2.js_divergence(a_before_sub, a_after_sub)
                    kl_neg, js_neg = DistanceMeasure2.js_divergence(b_before_sub, b_after_sub)

                    # todo: Hellinger Distance
                    hellinger_pos = DistanceMeasure2.hellinger_continous_1d(a_before_sub, a_after_sub)
                    hellinger_neg = DistanceMeasure2.hellinger_continous_1d(b_before_sub, b_after_sub)
                    # todo: wasserstein Distance
                    wasserstein_distance_pos = DistanceMeasure2.wasserstein_distance(a_before_sub, a_after_sub)
                    wasserstein_distance_neg = DistanceMeasure2.wasserstein_distance(b_before_sub, b_after_sub)

                    # todo: cramers_v Distance
                    #cramers_v_pos = DistanceMeasure.cramers_v(a_before_sub, a_after_sub)
                    #cramers_v_neg = DistanceMeasure.cramers_v(b_before_sub, b_after_sub)
                    # todo: total variation distance
                    total_variation_distance_pos = DistanceMeasure2.total_variation_distance(a_before_sub, a_after_sub)
                    total_variation_distance_neg = DistanceMeasure2.total_variation_distance(b_before_sub, b_after_sub)

                    kl_mean = np.mean([kl_pos, kl_neg])
                    js_mean = np.mean([js_pos, js_neg])
                    hellinger_mean = np.mean([hellinger_pos, hellinger_neg])
                    wasserstein_mean = np.mean([wasserstein_distance_pos, wasserstein_distance_neg])
                    #cramers_v_mean = np.mean([cramers_v_pos, cramers_v_neg])
                    total_variation_mean = np.mean([total_variation_distance_pos, total_variation_distance_neg])

                elif len(a_before_sub) >1:
                    kl_pos, js_pos = DistanceMeasure2.js_divergence(a_before_sub, a_after_sub)
                    hellinger_pos = DistanceMeasure2.hellinger_continous_1d(a_before_sub, a_after_sub)
                    wasserstein_distance_pos = DistanceMeasure2.wasserstein_distance(a_before_sub, a_after_sub)
                    #cramers_v_pos = DistanceMeasure.cramers_v(a_before_sub, a_after_sub)

                    total_variation_distance_pos = DistanceMeasure2.total_variation_distance(a_before_sub, a_after_sub)

                    kl_mean = kl_pos
                    js_mean = js_pos
                    hellinger_mean = hellinger_pos
                    wasserstein_mean = wasserstein_distance_pos
                    #cramers_v_mean = cramers_v_pos
                    total_variation_mean = total_variation_distance_pos
                elif len(b_before_sub) >1:
                    kl_neg, js_neg = DistanceMeasure2.js_divergence(b_before_sub, b_after_sub)
                    hellinger_neg = DistanceMeasure2.hellinger_continous_1d(b_before_sub, b_after_sub)
                    wasserstein_distance_neg = DistanceMeasure2.wasserstein_distance(b_before_sub, b_after_sub)
                    #cramers_v_neg = DistanceMeasure.cramers_v(b_before_sub, b_after_sub)
                    total_variation_distance_neg = DistanceMeasure2.total_variation_distance(b_before_sub, b_after_sub)
                    #import matplotlib.pyplot as plt
                    #fig, ax = plt.subplots(1, 1)
                    kl_mean = kl_neg
                    js_mean = js_neg
                    hellinger_mean = hellinger_neg
                    wasserstein_mean = wasserstein_distance_neg
                    #cramers_v_mean = cramers_v_neg
                    total_variation_mean = total_variation_distance_neg
                cramers_v_mean = DistanceMeasure.cramers_v(category_target_predicted_before,
                                                           category_target_predicted_after)
                #print('total_variation_mean and cramers_v_mean: ', total_variation_mean,cramers_v_mean, category_target_predicted_before, category_target_predicted_after)



                data_category[category] = {}
                data_category[category]['ACC'] = round(acc_category * 100 / len(category_target_original), 3)
                data_category[category]['TPR'] = TPR
                data_category[category]['FPR'] = FPR
                data_category[category]['distortion_multiple'] = distortion_multiple
                data_category[category]['distortion_feature'] = distortion_feature
                data_category[category]['hellinger_distortion'] = hellinger_distortion
                data_category[category]['wasserstein_distance_distortion'] = wasserstein_distance_distortion
                data_category[category]['cramers_v_distortion'] = cramers_v_distortion
                data_category[category]['total_variation_distortion'] = total_variation_distortion
                data_category[category]['kl_div_distortion'] = kl_div_distortion
                data_category[category]['js_div_distortion'] = js_div_distortion

                data_category[category]['SP'] = SP
                data_category[category]['ACC_after'] = round(acc_category_after * 100 / len(category_target_original),
                                                             3)
                data_category[category]['TPR_after'] = TPR_after
                data_category[category]['FPR_after'] = FPR_after
                data_category[category]['SP_after'] = SP_after
                data_category[category]['proportion'] = round(proportion * 100 / len(category_target_original), 3)
                data_category[category]['Importance'] = round(f_importance * 100 / len(category_target_original), 3)
                data_category[category]['hellinger_div'] = hellinger_mean
                data_category[category]['wasserstein_distance_div'] = wasserstein_mean
                data_category[category]['cramers_v_div'] = cramers_v_mean
                data_category[category]['total_variation_div'] = total_variation_mean

                data_category[category]['KL_Divergence'] = kl_mean
                data_category[category]['JS_Divergence'] = js_mean
        return data_category

    def fit_nn(self, iterations=10, swap_proportion=[0.1, 0.3, 0.5, 0.7], alpha=0.25):
        y_train = to_categorical(self.y_train)
        y_test = to_categorical(self.y_test)
        n_classes = y_test.shape[1]
        # NNmodel = NNClassifier(self.x_train.shape[1], n_classes)
        # NNmodel.fit(self.x_train, y_train) 0.01,
        #NNmodel = xgboost.XGBRegressor().fit(self.x_train, y_train)

        X, Y = split_features_target(self.data, index=self.target_index)
        #kf = model_selection.KFold(n_splits=10)
        kf = model_selection.StratifiedKFold(n_splits=5)
        #todo: cross-validation: https://medium.com/analytics-vidhya/cross-validation-with-code-in-python-55b342840089
        # fill the new kfold column
        for fold, (train_idx, test_idx) in enumerate(kf.split(X=self.df_data, y=Y)):
            train, test = self.df_data.loc[train_idx], self.df_data.loc[test_idx]
            train, test = train.to_numpy(), test.to_numpy()
            print(fold, 'train: ', train.shape, test.shape)
            self.x_train, self.y_train = split_features_target(train, self.target_index)
            self.x_test, self.y_test = split_features_target(test, self.target_index)
            y_train = to_categorical(self.y_train)
            y_test = to_categorical(self.y_test)
            NNmodel = Models2(self.x_train, self.x_test, y_train, y_test).decision_tree_regressor()

            p_pred = NNmodel.predict(self.x_test)
            a_before, b_before = [], []
            list_before, list_after = [], []
            for pred_ in p_pred:
                # print('predicted: ', np.around(pred_))
                a_before.append(pred_[0])
                b_before.append(pred_[1])

                arg_max = pred_.argmax(axis=-1)
                list_before.append(arg_max)

            N = round(len(self.x_test) / iterations)
            y_sampled_k = self.y_test
            for swap_proportion_ in swap_proportion:
                #for iter in range(iterations):
                #_, random_index_, _2, y_sampled_k = data_split(data=self.x_test, sample_size=swap_proportion)
                #random_index = random.choices(range(len(self.x_test)), k=N)
                #x_sampled_k = []
                #y_sampled_k = []
                #for random_i in random_index:
                #    #if i in random_index:
                #    x_sampled_k.append([x_ for x_ in self.x_test[random_i]])
                #    y_sampled_k.append(self.y_test[random_i])
                category_indices = []
                for column_id in range(self.x_train.shape[1]):
                    if is_categorical(self.x_test, column_id):
                        category_indices.append(column_id)
                for column_id in range(self.x_train.shape[1]):

                    x_sampled_k_feature = self.get_features_by_index(self.x_test, column_id)

                    # if column_id != self.sensitive_index:
                    # print('Column type name: ',self.data[0:, column_id].dtype.name)
                    a_after, b_after = [], []
                    if is_categorical(self.x_test, column_id):
                        _, x_test_altered = alter_feature_values_categorical_3(self.x_test, self.y_test,
                                                                          self.get_categorical_features(column_id),
                                                                          column_id, swap_proportion=swap_proportion_)
                        # q_pred = NNmodel.predict(tf.convert_to_tensor(x_test_altered, dtype=tf.float32))
                        q_pred = NNmodel.predict(x_test_altered)

                        #print('q_pred: ', q_pred)
                        list_after = []
                        for pred_ in q_pred:
                            #pred_ = abs(pred_)
                            a_after.append(pred_[0])
                            b_after.append(pred_[1])
                            arg_max = pred_.argmax(axis=-1)
                            #print('q_pred: ', pred_, pred_2, abs(pred_), abs(pred_[0]), abs(pred_[1]), arg_max)
                            list_after.append(arg_max)


                        x_test_altered_feature = self.get_features_by_index(x_test_altered, column_id)

                        distortion_multiple = DistanceMeasure2._distance_multiple(self.x_test,x_test_altered,category_indices, alpha=alpha)
                        distortion_feature = DistanceMeasure2._hamming_distance(x_sampled_k_feature,
                                                                                       x_test_altered_feature)
                        kl_div_distortion, js_div_distortion = DistanceMeasure2.js_divergence(x_sampled_k_feature,
                                                                                                 x_test_altered_feature)
                        hellinger_distortion = DistanceMeasure2.hellinger_continous_1d(x_sampled_k_feature,
                                                                                                 x_test_altered_feature)
                        wasserstein_distance_distortion =  DistanceMeasure2.wasserstein_distance(x_sampled_k_feature,
                                                                                                 x_test_altered_feature)
                        cramers_v_distortion = DistanceMeasure.cramers_v(x_sampled_k_feature,x_test_altered_feature)
                        total_variation_distortion = DistanceMeasure2.total_variation_distance(x_sampled_k_feature,
                                                                                                 x_test_altered_feature)

                        print('sampled k feature distortion values: ', kl_div_distortion, hellinger_distortion, wasserstein_distance_distortion, cramers_v_distortion, total_variation_distortion)

                        #kl_div_distortion, js_div_distortion = DistanceMeasure2.js_divergence_2d(x_sampled_k, x_test_altered)

                        #wasserstein_distance_distortion = DistanceMeasure2.wasserstein_distance_pdf2d(x_sampled_k, x_test_altered)

                        #hellinger_distortion = DistanceMeasure.hellinger_multivariate(x_sampled_k, x_test_altered)

                        #cramers_v_distortion = DistanceMeasure.cramers_v(x_sampled_k, x_test_altered)

                        #total_variation_distortion = DistanceMeasure2.total_variation_distance_2d(x_sampled_k, x_test_altered)
                        #print('cramers_v: ', cramers_v)
                        data_category_after = self.compute_metrics(list_before, list_after, column_id, p_pred,
                                                                   q_pred, distortion_multiple,distortion_feature, hellinger_distortion, wasserstein_distance_distortion,cramers_v_distortion, total_variation_distortion, kl_div_distortion, js_div_distortion, runkl=True)
                    else:
                        _, x_test_altered = alter_feature_value_continous_3(self.x_test, self.y_test, self.determine_range(column_id),
                                                                       column_id, swap_proportion=swap_proportion_)
                        # q_pred = NNmodel.predict(tf.convert_to_tensor(x_test_altered, dtype=tf.float32))
                        q_pred = NNmodel.predict(x_test_altered)
                        list_after = []
                        for pred_ in q_pred:
                            #pred_ = abs(pred_)
                            a_after.append(pred_[0])
                            b_after.append(pred_[1])
                            arg_max = pred_.argmax(axis=-1)
                            list_after.append(arg_max)
                        #x_sampled_k_feature = self.get_features_by_index(x_sampled_k, column_id)
                        x_test_altered_feature = self.get_features_by_index(x_test_altered, column_id)
                        distortion_multiple = DistanceMeasure2._distance_multiple(self.x_test,x_test_altered,category_indices, alpha=alpha)
                        distortion_feature = DistanceMeasure2._quared_error_proportion(x_sampled_k_feature,
                                                                                       x_test_altered_feature)
                        kl_div_distortion, js_div_distortion = DistanceMeasure2.js_divergence(x_sampled_k_feature,
                                                                                              x_test_altered_feature)
                        hellinger_distortion = DistanceMeasure2.hellinger_continous_1d(x_sampled_k_feature,
                                                                                       x_test_altered_feature)
                        wasserstein_distance_distortion = DistanceMeasure2.wasserstein_distance(x_sampled_k_feature,
                                                                                                x_test_altered_feature)
                        cramers_v_distortion = DistanceMeasure.cramers_v(x_sampled_k_feature, x_test_altered_feature)
                        total_variation_distortion = DistanceMeasure2.total_variation_distance(x_sampled_k_feature, x_test_altered_feature)


                        #kl_div_distortion, js_div_distortion = DistanceMeasure2.js_divergence_2d(x_sampled_k, x_test_altered)

                        #wasserstein_distance_distortion = DistanceMeasure2.wasserstein_distance_pdf2d(x_sampled_k, x_test_altered)
                        #hellinger_distortion = DistanceMeasure.hellinger_multivariate(x_sampled_k, x_test_altered)
                        #cramers_v_distortion = DistanceMeasure.cramers_v(x_sampled_k, x_test_altered)
                        #total_variation_distortion = DistanceMeasure2.total_variation_distance_2d(x_sampled_k,
                        #                                                                          x_test_altered)
                        #print('cramers_v: ', cramers_v)
                        data_category_after = self.compute_metrics_continous(list_before, list_after,
                                                                             column_id, p_pred, q_pred, distortion_multiple, distortion_feature, hellinger_distortion, wasserstein_distance_distortion,cramers_v_distortion, total_variation_distortion,kl_div_distortion, js_div_distortion, runkl=True)
                    for key, val in data_category_after.items():

                        self.data_writer2.writerow(
                            ['F_{}'.format(column_id), self.colums_list[column_id], key,swap_proportion_, data_category_after[key]['ACC'],
                             data_category_after[key]['ACC_after'], data_category_after[key]['distortion_multiple'],data_category_after[key]['distortion_feature'], data_category_after[key]['hellinger_distortion'],
                             data_category_after[key]['wasserstein_distance_distortion'],data_category_after[key]['cramers_v_distortion'], data_category_after[key]['total_variation_distortion'], data_category_after[key]['kl_div_distortion'],
                             data_category_after[key]['js_div_distortion'], data_category_after[key]['hellinger_div'], data_category_after[key]['wasserstein_distance_div'],
                             data_category_after[key]['cramers_v_div'], data_category_after[key]['total_variation_div'],
                             data_category_after[key]['KL_Divergence'],
                             data_category_after[key]['JS_Divergence'], data_category_after[key]['proportion'],
                             data_category_after[key]['Importance'], data_category_after[key]['SP'],
                             data_category_after[key]['SP_after'], data_category_after[key]['TPR'],
                             data_category_after[key]['TPR_after'], data_category_after[key]['FPR'],
                             data_category_after[key]['FPR_after']])



                    kl_mean, js_mean, hellinger_mean, wasserstein_mean, cramers_v_mean, total_variation_mean = 0, 0, 0, 0, 0, 0
                    if len(a_before) > 1 and len(b_before) > 0:
                        # todo: KL Divergence
                        kl_pos, js_pos = DistanceMeasure2.js_divergence(a_before, a_after)
                        kl_neg, js_neg = DistanceMeasure2.js_divergence(b_before, b_after)

                        # todo: Hellinger Distance
                        hellinger_pos = DistanceMeasure2.hellinger_continous_1d(a_before, a_after)
                        hellinger_neg = DistanceMeasure2.hellinger_continous_1d(b_before, b_after)
                        # todo: wasserstein Distance
                        wasserstein_distance_pos = DistanceMeasure2.wasserstein_distance(a_before, a_after)
                        wasserstein_distance_neg = DistanceMeasure2.wasserstein_distance(b_before, b_after)

                        # todo: cramers_v Distance
                        cramers_v_pos = DistanceMeasure.cramers_v(a_before, a_after)
                        cramers_v_neg = DistanceMeasure.cramers_v(b_before, b_after)

                        # todo: total variation distance
                        total_variation_distance_pos = DistanceMeasure2.total_variation_distance(a_before, a_after)
                        total_variation_distance_neg = DistanceMeasure2.total_variation_distance(b_before, b_after)

                        kl_mean = np.mean([kl_pos, kl_neg])
                        js_mean = np.mean([js_pos, js_neg])
                        hellinger_mean = np.mean([hellinger_pos, hellinger_neg])
                        wasserstein_mean = np.mean([wasserstein_distance_pos, wasserstein_distance_neg])
                        cramers_v_mean = np.mean([cramers_v_pos, cramers_v_neg])
                        total_variation_mean = np.mean([total_variation_distance_pos, total_variation_distance_neg])

                    elif len(a_before) > 1:
                        kl_pos, js_pos = DistanceMeasure2.js_divergence(a_before, a_after)
                        hellinger_pos = DistanceMeasure2.hellinger_continous_1d(a_before, a_after)
                        wasserstein_distance_pos = DistanceMeasure2.wasserstein_distance(a_before, a_after)
                        cramers_v_pos = DistanceMeasure.cramers_v(a_before, a_after)
                        total_variation_distance_pos = DistanceMeasure2.total_variation_distance(a_before, a_after)

                        kl_mean = kl_pos
                        js_mean = js_pos
                        hellinger_mean = hellinger_pos
                        wasserstein_mean = wasserstein_distance_pos
                        cramers_v_mean = cramers_v_pos
                        total_variation_mean = total_variation_distance_pos
                    elif len(b_before) > 1:
                        kl_neg, js_neg = DistanceMeasure2.js_divergence(b_before, b_after)
                        hellinger_neg = DistanceMeasure2.hellinger_continous_1d(b_before, b_after)
                        wasserstein_distance_neg = DistanceMeasure2.wasserstein_distance(b_before, b_after)
                        cramers_v_neg = DistanceMeasure.cramers_v(b_before, b_after)
                        total_variation_distance_neg = DistanceMeasure2.total_variation_distance(b_before, b_after)

                        kl_mean = kl_neg
                        js_mean = js_neg
                        hellinger_mean = hellinger_neg
                        wasserstein_mean = wasserstein_distance_neg
                        cramers_v_mean = cramers_v_neg
                        total_variation_mean = total_variation_distance_neg


                    #print('hellinger', hellinger_pos, hellinger_neg, 'wasserstein: ', wasserstein_distance_pos, wasserstein_distance_neg, kl_pos, js_pos)


                    # self.data_writer2.writerow(
                    #    ['F_{}'.format(column_id), self.colums_list[column_id], key, 'Combined', '',
                    #     kl_div_positive, kl_div_negative, '',
                    #     '', '', ''])

                    #print('KL Divergence {} for positive class: '.format(column_id), kl_div_positive)
                    #print('KL Divergence {} for negative class: '.format(column_id), kl_div_negative)

                    if self.log_data:
                        TP = 0
                        FP = 0
                        TN = 0
                        FN = 0
                        TP_after = 0
                        FP_after = 0
                        TN_after = 0
                        FN_after = 0
                        list_before2 = []
                        for i in range(len(y_test)):
                            # print(y_predicted[i], len(self.y_test[i]), self.y_test[i], self.y_test)
                            list_before2.append(list_before[i])
                            if list_before[i] == 1 and self.y_test[i] == 1:
                                TP += 1
                            if list_before[i] == 1 and self.y_test[i] == 0:
                                FP += 1
                            if list_before[i] == 0 and self.y_test[i] == 0:
                                TN += 1
                            if list_before[i] == 0 and self.y_test[i] == 1:
                                FN += 1
                            if list_after[i] == 1 and self.y_test[i] == 1:
                                TP_after += 1
                            if list_after[i] == 1 and self.y_test[i] == 0:
                                FP_after += 1
                            if list_after[i] == 0 and self.y_test[i] == 0:
                                TN_after += 1
                            if list_after[i] == 0 and self.y_test[i] == 1:
                                FN_after += 1

                        # category_target_original = np.array(category_target_original)
                        list_before = np.array(list_before)
                        list_before2 = np.array(list_before2)
                        list_after = np.array(list_after)
                        y_test_ = self.y_test # [self.y_test[i] for i in range(len(random_index))]
                        acc_category = round((np.sum(y_test_ == list_before) * 100 / len(y_test_)), 3)
                        TPR = calculate_TPR(TP, FP, TN, FN)
                        FPR = calculate_FPR(TP, FP, TN, FN)
                        SP = calculate_SP(TP, FP, TN, FN)

                        casuality = round((np.sum(list_before2 != list_after) * 100 / len(y_test_)), 3)

                        acc_category_after = round((np.sum(y_sampled_k == list_after) * 100 / len(y_sampled_k)), 3)
                        f_importance = round((np.sum(y_sampled_k != list_after) * 100 / len(y_sampled_k)), 3)
                        TPR_after = calculate_TPR(TP_after, FP_after, TN_after, FN_after)
                        FPR_after = calculate_FPR(TP_after, FP_after, TN_after, FN_after)
                        SP_after = calculate_SP(TP_after, FP_after, TN_after, FN_after)

                        self.data_writer.writerow(
                            ['F_{}'.format(column_id), self.colums_list[column_id],swap_proportion_, acc_category, acc_category_after, distortion_multiple, distortion_feature,
                             hellinger_distortion, wasserstein_distance_distortion, cramers_v_distortion, total_variation_distortion, kl_div_distortion, js_div_distortion,
                             hellinger_mean, wasserstein_mean, cramers_v_mean, total_variation_mean,
                             kl_mean, js_mean, casuality, f_importance, SP, SP_after, TPR, TPR_after, FPR,
                             FPR_after])


        if self.log_data:
            self.data_file.close()


if __name__ == '__main__':
    path = '../dataset/'
    ### Adult dataset
    # target_column = 'Probability'
    correlation_threshold = 0.45  # 0.35
    loadData = LoadData(path, threshold=correlation_threshold)  # ,threshold=correlation_threshold
    data_name = 'adult-45_v2_50_'  # _35_threshold

    df_adult = loadData.load_adult_data('adult.data.csv')
    # df_adult = loadData.load_credit_defaulter('default_of_credit_card_clients.csv')
    #df_adult = loadData.load_clevelan_heart_data('processed.cleveland.data.csv')
    #df_adult = loadData.load_student_data('Student.csv')
    #compas-scores-two-years
    #df_adult = loadData.load_compas_data('compas-scores-two-years.csv')
    sensitive_list = loadData.sensitive_list
    sensitive_indices = loadData.sensitive_indices
    colums_list = df_adult.columns.tolist()

    #data_name = 'clevelan_heart'
    #data_name = 'compas'
    #data_name = 'Student'
    data_name = 'adult'
    path = '../dataset/experiments/data/'
    load_data = LoadTrainTest(path)
    train, test, target_index = load_data.load_adult(data_name)

    # corr_columns = get_correlation(df_adult, correlation_threshold)

    # corrMatrix = df_adult.corr()
    # sn.heatmap(corrMatrix, annot=True)
    # corr = df_adult.corr()
    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.matshow(corr)
    # plt.xticks(range(len(corr.columns)), corr.columns)
    # plt.yticks(range(len(corr.columns)), corr.columns)

    # plt.savefig(path + "/png/{}.png".format(data_name))
    # plt.show()
    target_name = loadData.target_name
    target_index = loadData.target_index
    for colum in colums_list:
        print(colum, df_adult[target_name].corr(df_adult[colum]))
    # corr = df_adult.corr()
    # corr.style.background_gradient(cmap='coolwarm')

    df_adult.to_csv(path + '{}-transformed.csv'.format(data_name), index=False)

    # print(df_adult)
    # print(target_name)
    # print(target_index)
    sensitivityAnalysis = SensitivityAnalysis(df_adult.to_numpy(), df_adult, target_index=target_index,
                                              sensitive_name=sensitive_list[0], sensitive_index=sensitive_indices[0],
                                              data_name=data_name, colums_list=colums_list,
                                              threshold=correlation_threshold)
    sensitivityAnalysis.fit_nn()

    '''
    rank_path = '../dataset/experiments/data/{}/sensitivity/'.format(data_name)
    if not os.path.exists(rank_path + "global"):
        os.makedirs(rank_path + "global")
    if not os.path.exists(rank_path + "local"):
        os.makedirs(rank_path + "local")
    if not os.path.exists(rank_path + "logs"):
        os.makedirs(rank_path + "logs")
    rank_path_global = rank_path + "global/"
    rank_path_local = rank_path + "local/"
    rank_path_log = rank_path + "logs/"
    #self.new_path = log_path + str(data_name) + "/{}/".format(id)
    #self.log_path_global = log_path + 'KL_Divergence_global_{}.csv'.format(data_name)
    data_file_global = open(rank_path_global+'log_{}.csv'.format(data_name), mode='w', newline='', encoding='utf-8')
    data_writer_global = csv.writer(data_file_global)
    data_writer_global.writerow(['Feature', 'Rank_Global_AVG','Rank_Global_MSF', 'Rank_Local_AVG', 'Rank_Local_MSF'])

    ## todo: compute ranking
    df_global = pd.read_csv(sensitivityAnalysis.log_path_global)
    ranking_attributes = ['JS_Divergence', 'Casuality', 'Importance']  # , 'SP', 'Casuality',
    # ranking_attributes = ['JS_Divergence', 'Casuality']  # , 'SP', 'Casuality',
    PSA = 'Feature'
    sub_category = 'Category'
    ranking = Ranking()
    rank_global_psa, rank_global, rank_global_median = ranking.rank_average(df_global, feature_name=PSA, PSA=ranking_attributes)
    rank_global = ranking.sort_dict(rank_global, reverse=False)
    print('Global ranking: ', rank_global)






    # df = pd.read_csv(path + 'logging/log_kl_divergence_adult-45.csv')
    df_local = pd.read_csv(sensitivityAnalysis.log_path_local)
    rank_local_psa, rank_local, rank_local_median = ranking.rank_average(df_local, PSA, sub_category=sub_category, PSA=ranking_attributes)
    rank_local = ranking.sort_dict(rank_local, reverse=False)
    data_ranks = {}
    for key, val in rank_local.items():
        key_split = key.split('|')[0]
        if key_split in data_ranks.keys():
            data_ranks[key_split].append(val)
        else:
            data_ranks[key_split] = [val]
    for key, val in data_ranks.items():
        data_ranks[key] = np.mean(val)

    print('Local ranking: ', rank_local)
    print('Local ranking averaging: ', ranking.sort_dict(data_ranks, reverse=False))
    rank = 0
    for key, val in data_ranks.items():
        if rank != val:
            rank += 1
        data_ranks[key] = rank

    #data_ranks = ranking.sort_dict(data_ranks, reverse=)


    ## Rank by multiplicative score function
    print('\n***** Ranking by Multiplicative Score Function ********\n')
    rank_global_psa, rank_global_2, rank_global_median = ranking.rank_multiplicative_score_function(df_global, feature_name=PSA, PSA=ranking_attributes)
    rank_global = ranking.sort_dict(rank_global, reverse=False)
    print('Global ranking MSF: ', rank_global_2)

    rank_local_psa, rank_local, rank_local_median = ranking.rank_multiplicative_score_function(df_local, PSA, sub_category=sub_category,
                                                            PSA=ranking_attributes)
    rank_local = ranking.sort_dict(rank_local, reverse=False)
    data_ranks_2 = {}
    for key, val in rank_local.items():
        key_split = key.split('|')[0]
        if key_split in data_ranks_2.keys():
            data_ranks_2[key_split].append(val)
        else:
            data_ranks_2[key_split] = [val]
    for key, val in data_ranks_2.items():
        data_ranks_2[key] = np.mean(val)
    data_ranks_2 = ranking.sort_dict(data_ranks_2, reverse=True)

    rank_id = 0
    for key, val in data_ranks_2.items():
        if val != rank_id:
            rank_id += 1
        data_ranks_2[key] = rank_id


    print('Local ranking: ', rank_local)
    print('Local ranking MSF: ', ranking.sort_dict(data_ranks_2, reverse=False))

    for key, val in rank_global.items():
        data_writer_global.writerow([key, val, rank_global_2.get(key), data_ranks.get(key), data_ranks_2.get(key)])
    #data_writer_global.writerow(['Feature', 'Rank_Global_AVG', 'Rank_Global_MSF', 'Rank_Local_AVG', 'Rank_Local_MSF'])
    data_file_global.close()'''
