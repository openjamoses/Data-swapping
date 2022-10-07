import csv
import os
import random

import shap
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
#from sklearn.model_selection import KFold

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
        self.log_path_global = log_path + 'Divergence_2_columns_{}.csv'.format(data_name)
        self.data_file = open(self.log_path_global, mode='w', newline='',
                              encoding='utf-8')
        self.data_writer = csv.writer(self.data_file)
        self.data_writer.writerow(
            ['ID','ID2', 'Feature', 'Feature2','swap_proportion', 'Type', 'Acc', 'Acc_after', 'distortion_multiple', 'distortion_feature', 'hellinger_div','wasserstein_div','cramers_v_div', 'total_variation_div', 'KL_div', 'JS_Div', 'Casuality', 'Importance','effect_distance', 'effect_distance_contrained', 'SP', 'SP_after',
             'TPR', 'TPR_after', 'FPR', 'FPR_after'])

        self.log_path_local = log_path + 'Divergence_2_columns_local_{}.csv'.format(data_name)
        self.data_file_2 = open(self.log_path_local, mode='w', newline='',
                              encoding='utf-8')
        self.data_writer2 = csv.writer(self.data_file_2)
        self.data_writer2.writerow(
            ['ID', 'ID2', 'Feature', 'Feature2','Category1', 'Category2', 'swap_proportion', 'Type', 'Acc', 'Acc_after', 'distortion_multiple', 'distortion_feature', 'hellinger_div', 'wasserstein_div', 'cramers_v_div', 'total_variation_div', 'KL_div',
             'JS_Div', 'Casuality', 'Importance', 'effect_distance', 'effect_distance_contrained', 'SP', 'SP_after',
             'TPR', 'TPR_after', 'FPR', 'FPR_after'])

        self.log_path_shap = log_path + 'Shap_f_importance_{}.csv'.format(data_name)
        self.data_file_3 = open(self.log_path_shap, mode='w', newline='',
                                encoding='utf-8')
        self.data_writer3 = csv.writer(self.data_file_3)
        self.data_writer3.writerow(['Fold', 'swap_proportion','Feature', 'Shap', 'Importance'])

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
        print('column data: ', set(self.data[0:, feature_index]))
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
        #print(data_range)
        #percentile_50_ = (min(data_range) + max(data_range))/2 #np.percentile(list(set(data_range)), 50)
        percentile_50_ = np.percentile(list(set(data_range)), 50)
        # percentile_50_ = np.mean(data_range)
        percentile_75 = np.percentile(data_range, 75)
        percentile_50 = max([i for i in list(set(data_range)) if i <= percentile_50_])
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
        #print('Data Bining: ', feature_index)
        # for key, val in folded_data.items():
        #     print(key, len(val), val)
        return folded_data  # list(folded_data.values())
    def get_posible_values_2_columns(self, feature_indices=[0,1]):

        #f1 = ''
        #f2 = ''
        if is_categorical(self.x_test, feature_indices[0]):
            category_data = self.determine_range(feature_indices[0])
            #f1 = 'Category'
        else:
            category_data = self.determine_range(feature_indices[0])
            #f1 = 'Continous'
        if is_categorical(self.x_test, feature_indices[1]):
            category_data_2 = self.determine_range(feature_indices[1])
            #f2 = 'Category'
        else:
            category_data_2 = self.determine_range(feature_indices[1])
            #f2 = 'Continous'
        posible_values = {}
        posible_values[feature_indices[0]] = category_data
        posible_values[feature_indices[1]] = category_data_2

        #print('Feature Descriptions: ', f1, f2, category_data, category_data_2)
        #print(f1, category_data)
        #print(f2, category_data_2)
        return posible_values
    def compute_metrics_direct(self, perc_parity, x_test_altered, random_index_, category_indices, y_predicted_before, y_predicted_after, column_id, p_pred,
                        q_pred, distortion_multiple,distortion_feature, runkl=False):

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
                cont_swap = True
                if i in random_index_ and perc_parity != None:
                    distortion_value = DistanceMeasure2._distance_single(self.x_test[i], x_test_altered[i], category_indices)
                    if distortion_value > perc_parity:
                        cont_swap = False
                if self.x_test[i][column_id] == category and cont_swap:
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
            impont = 0
            if len(category_target_original) > 0:
                impont = round(f_importance * 100 / len(category_target_original), 3)
            #if str(impont) == 'nan' or len(category_target_original) == 0:
            #    print('nan found! ',category, category_list, column_id,  category_target_original)
            proportion = np.sum(
                category_target_predicted_before != category_target_predicted_after)  # calculate_proportion(TP, FP, TN, FN)

            kl_mean, js_mean, hellinger_mean, wasserstein_mean, cramers_v_mean, total_variation_mean, effect_distance_mean, effect_distance_contrained_mean = 0, 0, 0, 0, 0, 0, 0, 0
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

                effect_distance_pos, effect_distance_contrained_pos = DistanceMeasure2.effect_distance(a_before_sub, a_after_sub, fair_error)
                effect_distance_neg, effect_distance_contrained_neg = DistanceMeasure2.effect_distance(b_before_sub, b_after_sub,
                                                                                       fair_error)
                kl_mean = np.mean([kl_pos, kl_neg])
                js_mean = np.mean([js_pos, js_neg])

                effect_distance_mean = np.mean([effect_distance_pos, effect_distance_neg])
                effect_distance_contrained_mean = np.mean([effect_distance_contrained_pos, effect_distance_contrained_neg])


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
                effect_distance_pos, effect_distance_contrained_pos = DistanceMeasure2.effect_distance(a_before_sub,
                                                                                                       a_after_sub,
                                                                                                       fair_error)

                kl_mean = kl_pos
                js_mean = js_pos
                effect_distance_mean = effect_distance_pos
                effect_distance_contrained_mean = effect_distance_contrained_pos

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
                effect_distance_neg, effect_distance_contrained_neg = DistanceMeasure2.effect_distance(b_before_sub,
                                                                                                       b_after_sub,
                                                                                                       fair_error)

                kl_mean = kl_neg
                js_mean = js_neg

                effect_distance_mean = effect_distance_neg
                effect_distance_contrained_mean = effect_distance_contrained_neg

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
                data_category[category]['SP'] = SP
                data_category[category]['ACC_after'] = round(acc_category_after * 100 / len(category_target_original), 3)
                data_category[category]['TPR_after'] = TPR_after
                data_category[category]['FPR_after'] = FPR_after
                data_category[category]['SP_after'] = SP_after
                data_category[category]['proportion'] = round(proportion * 100 / len(category_target_original), 3)
                data_category[category]['Importance'] = round(f_importance * 100 / len(category_target_original), 3)
                data_category[category]['distortion_multiple'] = distortion_multiple
                data_category[category]['distortion_feature'] = distortion_feature

                data_category[category]['hellinger_div'] = hellinger_mean
                data_category[category]['wasserstein_distance_div'] = wasserstein_mean
                data_category[category]['cramers_v_div'] = cramers_v_mean
                data_category[category]['total_variation_div'] = total_variation_mean

                data_category[category]['effect_distance'] = effect_distance_mean
                data_category[category]['effect_distance_contrained'] = effect_distance_contrained_mean

                data_category[category]['KL_Divergence'] = kl_mean
                data_category[category]['JS_Divergence'] = js_mean

                #'worse_case_unfair', 'epsilon_unfair',

            #print('NAN category_target_original: ', acc_category_after, round(f_importance * 100 / len(category_target_original), 3), category_target_original,
            #      category_target_predicted_after)
        return data_category

    def compute_metrics_continous_direct(self, perc_parity, x_test_altered, random_index_, category_indices, y_predicted_before, y_predicted_after, column_id, p_pred,
                                  q_pred,distortion_multiple,distortion_feature, runkl=False):
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
                cont_swap = True

                if i in random_index_ and perc_parity != None:
                    distortion_value = DistanceMeasure2._distance_single(self.x_test[i], x_test_altered[i], category_indices)
                    if distortion_value > perc_parity:
                        cont_swap = False
                    #x_test_altered_row = x_test_altered[i]
                    #distortion_value = DistanceMeasure2._distance_single(self.x_test[i], x_test_altered[i])
                if self.x_test[i][column_id] in val and cont_swap:
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
                kl_mean, js_mean, hellinger_mean, wasserstein_mean, cramers_v_mean, total_variation_mean, effect_distance_mean, effect_distance_contrained_mean = 0, 0, 0, 0, 0, 0, 0, 0
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

                    effect_distance_pos, effect_distance_contrained_pos = DistanceMeasure2.effect_distance(a_before_sub,
                                                                                                           a_after_sub,
                                                                                                           fair_error)
                    effect_distance_neg, effect_distance_contrained_neg = DistanceMeasure2.effect_distance(b_before_sub,
                                                                                                           b_after_sub,
                                                                                                           fair_error)
                    effect_distance_mean = np.mean([effect_distance_pos, effect_distance_neg])
                    effect_distance_contrained_mean = np.mean(
                        [effect_distance_contrained_pos, effect_distance_contrained_neg])

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
                    effect_distance_pos, effect_distance_contrained_pos = DistanceMeasure2.effect_distance(a_before_sub,
                                                                                                           a_after_sub,
                                                                                                           fair_error)

                    effect_distance_mean = effect_distance_pos
                    effect_distance_contrained_mean = effect_distance_contrained_pos
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
                    effect_distance_neg, effect_distance_contrained_neg = DistanceMeasure2.effect_distance(b_before_sub,
                                                                                                           b_after_sub,
                                                                                                           fair_error)

                    effect_distance_mean = effect_distance_neg
                    effect_distance_contrained_mean = effect_distance_contrained_neg

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
                data_category[category]['effect_distance'] = effect_distance_mean
                data_category[category]['effect_distance_contrained'] = effect_distance_contrained_mean

                data_category[category]['KL_Divergence'] = kl_mean
                data_category[category]['JS_Divergence'] = js_mean
        return data_category
    def compute_(self, data_category, perc_parity, x_test_altered, random_index_, category_indices, y_predicted_before, y_predicted_after, p_pred, q_pred, column_id, column_id2, val, val2, category_1, category_2, distortion_multiple, distortion_feature):
        category_target_original = []
        category_target_predicted_before = []
        category_target_predicted_after = []

        category_target_original_indirect = []
        category_target_predicted_before_indirect = []
        category_target_predicted_after_indirect = []
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

        a_before_sub_indirect, b_before_sub_indirect = [], []
        a_after_sub_indirect, b_after_sub_indirect = [], []

        x_test_modified, x_test_modified2 = [], []



        for i in range(len(self.x_test)):
            cont_swap = True
            if i in random_index_ and perc_parity != None:
                #if perc_parity != None:
                distortion_value = DistanceMeasure2._distance_single(self.x_test[i], x_test_altered[i],
                                                                     category_indices)
                if distortion_value > perc_parity:
                    cont_swap = False
                # x_test_altered_row = x_test_altered[i]
                # distortion_value = DistanceMeasure2._distance_single(self.x_test[i], x_test_altered[i])
            if (self.x_test[i][column_id] in val) and (self.x_test[i][column_id2] in val2) and cont_swap:
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
                # if runkl:
                a_before_sub.append(p_pred[i][0])
                b_before_sub.append(p_pred[i][1])
                # if q_pred != None:
                a_after_sub.append(q_pred[i][0])
                b_after_sub.append(q_pred[i][1])
                x_test_modified.append(self.x_test[i])
            elif (not self.x_test[i][column_id] in val) and (self.x_test[i][column_id2] in val2) and cont_swap:
                a_before_sub_indirect.append(p_pred[i][0])
                b_before_sub_indirect.append(p_pred[i][1])
                # if q_pred != None:
                a_after_sub_indirect.append(q_pred[i][0])
                b_after_sub_indirect.append(q_pred[i][1])
                x_test_modified.append(self.x_test[i])

                category_target_original_indirect.append(self.y_test[i])
                category_target_predicted_before_indirect.append(y_predicted_before[i])
                category_target_predicted_after_indirect.append(y_predicted_after[i])

        category_target_original = np.array(category_target_original)
        category_target_predicted_before = np.array(category_target_predicted_before)
        category_target_predicted_after = np.array(category_target_predicted_after)
        acc_category = np.sum(category_target_original == category_target_predicted_before)
        TPR = calculate_TPR(TP, FP, TN, FN)
        FPR = calculate_FPR(TP, FP, TN, FN)
        SP = calculate_SP(TP, FP, TN, FN)

        acc_category_after = np.sum(category_target_original == category_target_predicted_after)
        # if str(acc_category_after) == 'nan':
        f_importance = np.sum(category_target_original != category_target_predicted_after)
        TPR_after = calculate_TPR(TP_after, FP_after, TN_after, FN_after)
        FPR_after = calculate_FPR(TP_after, FP_after, TN_after, FN_after)
        SP_after = calculate_SP(TP_after, FP_after, TN_after, FN_after)

        # if len(category_target_original) == 0:
        #    print('f_importance: ', category, f_importance, category_target_original, category_target_predicted_after)

        proportion = np.sum(
            category_target_predicted_before != category_target_predicted_after)  # calculate_proportion(TP, FP, TN, FN)
        kl_pos = None
        kl_neg = None
        #print('checking distances error starts..')

        #print(b_before_sub, a_before_sub)
        if len(val) > 0 and len(category_target_predicted_before) > 0:
            IMPACT_TYPE = 'NDI'
            # print('before call: ', a_before_sub, a_after_sub)
            #print('checking starts..')
            kl_mean, js_mean, hellinger_mean, wasserstein_mean, cramers_v_mean, total_variation_mean, effect_distance_mean, effect_distance_contrained_mean = 0, 0, 0, 0, 0, 0, 0, 0
            if len(a_before_sub) > 1 and len(b_before_sub) > 0:
                # todo: KL Divergence
               #print('checking mean 1 starts..')
                #print('checking kl_pos error starts..')
                kl_pos, js_pos = DistanceMeasure2.js_divergence(a_before_sub, a_after_sub)
                kl_neg, js_neg = DistanceMeasure2.js_divergence(b_before_sub, b_after_sub)
                #print('checking kl_pos error ends..')
                # todo: Hellinger Distance
                #print('checking Hellinger error starts..')
                #print('Hellinger distance starts ----')
                hellinger_pos = DistanceMeasure2.hellinger_continous_1d(a_before_sub, a_after_sub)
                hellinger_neg = DistanceMeasure2.hellinger_continous_1d(b_before_sub, b_after_sub)
                #print('checking Hellinger error ends..')
                #print('Hellinger distance ends ----')
                # todo: wasserstein Distance
                #print('checking wasserstein error starts..')
                wasserstein_distance_pos = DistanceMeasure2.wasserstein_distance(a_before_sub, a_after_sub)
                wasserstein_distance_neg = DistanceMeasure2.wasserstein_distance(b_before_sub, b_after_sub)
                #print('checking wasserstein error ends..')
                # todo: cramers_v Distance
                # cramers_v_pos = DistanceMeasure.cramers_v(a_before_sub, a_after_sub)
                # cramers_v_neg = DistanceMeasure.cramers_v(b_before_sub, b_after_sub)
                # todo: total variation distance
                #print('checking total_variation_distance_pos error starts..')
                total_variation_distance_pos = DistanceMeasure2.total_variation_distance(a_before_sub, a_after_sub)
                total_variation_distance_neg = DistanceMeasure2.total_variation_distance(b_before_sub, b_after_sub)
                #print('checking total_variation_distance_pos error ends..')


                #print('checking effect_distance_pos error starts..')
                effect_distance_pos, effect_distance_contrained_pos = DistanceMeasure2.effect_distance(a_before_sub,
                                                                                                       a_after_sub,
                                                                                                       fair_error)
                effect_distance_neg, effect_distance_contrained_neg = DistanceMeasure2.effect_distance(b_before_sub,
                                                                                                       b_after_sub,
                                                                                                       fair_error)
                effect_distance_mean = np.mean([effect_distance_pos, effect_distance_neg])
                effect_distance_contrained_mean = np.mean(
                    [effect_distance_contrained_pos, effect_distance_contrained_neg])

                #print('checking effect_distance_pos error end..')


                kl_mean = np.mean([kl_pos, kl_neg])
                js_mean = np.mean([js_pos, js_neg])
                hellinger_mean = np.mean([hellinger_pos, hellinger_neg])
                wasserstein_mean = np.mean([wasserstein_distance_pos, wasserstein_distance_neg])
                # cramers_v_mean = np.mean([cramers_v_pos, cramers_v_neg])
                total_variation_mean = np.mean([total_variation_distance_pos, total_variation_distance_neg])
                #print('checking mean 1 end..')
            elif len(a_before_sub) > 1:
                #print('checking mean 2 starts..')
                kl_pos, js_pos = DistanceMeasure2.js_divergence(a_before_sub, a_after_sub)
                hellinger_pos = DistanceMeasure2.hellinger_continous_1d(a_before_sub, a_after_sub)
                wasserstein_distance_pos = DistanceMeasure2.wasserstein_distance(a_before_sub, a_after_sub)
                # cramers_v_pos = DistanceMeasure.cramers_v(a_before_sub, a_after_sub)

                total_variation_distance_pos = DistanceMeasure2.total_variation_distance(a_before_sub, a_after_sub)
                effect_distance_pos, effect_distance_contrained_pos = DistanceMeasure2.effect_distance(a_before_sub,
                                                                                                       a_after_sub,
                                                                                                       fair_error)


                effect_distance_mean = effect_distance_pos
                effect_distance_contrained_mean = effect_distance_contrained_pos
                kl_mean = kl_pos
                js_mean = js_pos
                hellinger_mean = hellinger_pos
                wasserstein_mean = wasserstein_distance_pos
                # cramers_v_mean = cramers_v_pos
                total_variation_mean = total_variation_distance_pos
                #print('checking mean 2 ends..')
            elif len(b_before_sub) > 1:
                #print('checking mean 3 starts..')
                kl_neg, js_neg = DistanceMeasure2.js_divergence(b_before_sub, b_after_sub)
                hellinger_neg = DistanceMeasure2.hellinger_continous_1d(b_before_sub, b_after_sub)
                wasserstein_distance_neg = DistanceMeasure2.wasserstein_distance(b_before_sub, b_after_sub)
                # cramers_v_neg = DistanceMeasure.cramers_v(b_before_sub, b_after_sub)
                total_variation_distance_neg = DistanceMeasure2.total_variation_distance(b_before_sub, b_after_sub)
                # import matplotlib.pyplot as plt
                # fig, ax = plt.subplots(1, 1)
                effect_distance_neg, effect_distance_contrained_neg = DistanceMeasure2.effect_distance(b_before_sub,
                                                                                                       b_after_sub,
                                                                                                       fair_error)

                effect_distance_mean = effect_distance_neg
                effect_distance_contrained_mean = effect_distance_contrained_neg

                kl_mean = kl_neg
                js_mean = js_neg
                hellinger_mean = hellinger_neg
                wasserstein_mean = wasserstein_distance_neg
                # cramers_v_mean = cramers_v_neg
                total_variation_mean = total_variation_distance_neg
                #print('checking mean 3 ends..')
            #print('checking cramers_v_mean error ends..')
            #print('checking 1 starts..')
            #print(hellinger_mean, wasserstein_mean, total_variation_mean, js_mean)
            cramers_v_mean = DistanceMeasure.cramers_v(category_target_predicted_before,
                                                       category_target_predicted_after)

            #print('checking 1 ends..')

            # print('total_variation_mean and cramers_v_mean: ', total_variation_mean,cramers_v_mean, category_target_predicted_before, category_target_predicted_after)
            #print('checking 2 starts..')
            if not category_1 in data_category.keys():
                data_category[category_1] = {}
            if not category_2 in data_category[category_1].keys():
                data_category[category_1][category_2] = {}
                #if category_2 in data_category[category_1].keys():
            data_category[category_1][category_2][IMPACT_TYPE] = {}
            data_category[category_1][category_2][IMPACT_TYPE]['ACC'] = round(acc_category * 100 / len(category_target_original), 3)
            data_category[category_1][category_2][IMPACT_TYPE]['TPR'] = TPR
            data_category[category_1][category_2][IMPACT_TYPE]['FPR'] = FPR
            data_category[category_1][category_2][IMPACT_TYPE]['distortion_multiple'] = distortion_multiple
            data_category[category_1][category_2][IMPACT_TYPE]['distortion_feature'] = distortion_feature

            data_category[category_1][category_2][IMPACT_TYPE]['SP'] = SP
            data_category[category_1][category_2][IMPACT_TYPE]['ACC_after'] = round(acc_category_after * 100 / len(category_target_original),
                                                         3)
            data_category[category_1][category_2][IMPACT_TYPE]['TPR_after'] = TPR_after
            data_category[category_1][category_2][IMPACT_TYPE]['FPR_after'] = FPR_after
            data_category[category_1][category_2][IMPACT_TYPE]['SP_after'] = SP_after
            data_category[category_1][category_2][IMPACT_TYPE]['proportion'] = round(proportion * 100 / len(category_target_original), 3)
            data_category[category_1][category_2][IMPACT_TYPE]['Importance'] = round(f_importance * 100 / len(category_target_original), 3)
            data_category[category_1][category_2][IMPACT_TYPE]['hellinger_div'] = hellinger_mean
            data_category[category_1][category_2][IMPACT_TYPE]['wasserstein_distance_div'] = wasserstein_mean
            data_category[category_1][category_2][IMPACT_TYPE]['cramers_v_div'] = cramers_v_mean
            data_category[category_1][category_2][IMPACT_TYPE]['total_variation_div'] = total_variation_mean
            data_category[category_1][category_2][IMPACT_TYPE]['effect_distance'] = effect_distance_mean
            data_category[category_1][category_2][IMPACT_TYPE]['effect_distance_contrained'] = effect_distance_contrained_mean

            data_category[category_1][category_2][IMPACT_TYPE]['KL_Divergence'] = kl_mean
            data_category[category_1][category_2][IMPACT_TYPE]['JS_Divergence'] = js_mean

        if len(val) > 0 and len(category_target_predicted_before_indirect) > 0:
            f_importance = np.sum(category_target_original_indirect != category_target_predicted_after_indirect)
            proportion = np.sum(
                category_target_predicted_before_indirect != category_target_predicted_after_indirect)  # calculate_proportion(TP, FP, TN, FN)
            kl_pos = None
            kl_neg = None

            IMPACT_TYPE = 'NII'
            # print('before call: ', a_before_sub, a_after_sub)
            # print('checking starts..')
            kl_mean, js_mean, hellinger_mean, wasserstein_mean, cramers_v_mean, total_variation_mean, effect_distance_mean, effect_distance_contrained_mean = 0, 0, 0, 0, 0, 0, 0, 0
            if len(a_before_sub) > 1 and len(b_before_sub) > 0:
                # todo: KL Divergence
                # print('checking mean 1 starts..')
                # print('checking kl_pos error starts..')
                kl_pos, js_pos = DistanceMeasure2.js_divergence(a_before_sub_indirect, a_after_sub_indirect)
                kl_neg, js_neg = DistanceMeasure2.js_divergence(b_before_sub_indirect, b_after_sub_indirect)
                # print('checking kl_pos error ends..')
                # todo: Hellinger Distance
                # print('checking Hellinger error starts..')
                # print('Hellinger distance starts ----')
                hellinger_pos = DistanceMeasure2.hellinger_continous_1d(a_before_sub_indirect, a_after_sub_indirect)
                hellinger_neg = DistanceMeasure2.hellinger_continous_1d(b_before_sub_indirect, b_after_sub_indirect)
                # print('checking Hellinger error ends..')
                # print('Hellinger distance ends ----')
                # todo: wasserstein Distance
                # print('checking wasserstein error starts..')
                wasserstein_distance_pos = DistanceMeasure2.wasserstein_distance(a_before_sub_indirect, a_after_sub_indirect)
                wasserstein_distance_neg = DistanceMeasure2.wasserstein_distance(b_before_sub_indirect, b_after_sub_indirect)
                # print('checking wasserstein error ends..')
                # todo: cramers_v Distance
                # cramers_v_pos = DistanceMeasure.cramers_v(a_before_sub, a_after_sub)
                # cramers_v_neg = DistanceMeasure.cramers_v(b_before_sub, b_after_sub)
                # todo: total variation distance
                # print('checking total_variation_distance_pos error starts..')
                total_variation_distance_pos = DistanceMeasure2.total_variation_distance(a_before_sub_indirect, a_after_sub_indirect)
                total_variation_distance_neg = DistanceMeasure2.total_variation_distance(b_before_sub_indirect, b_after_sub_indirect)
                # print('checking total_variation_distance_pos error ends..')

                # print('checking effect_distance_pos error starts..')
                effect_distance_pos, effect_distance_contrained_pos = DistanceMeasure2.effect_distance(a_before_sub_indirect, a_after_sub_indirect,
                                                                                                       fair_error)
                effect_distance_neg, effect_distance_contrained_neg = DistanceMeasure2.effect_distance(b_before_sub_indirect, b_after_sub_indirect,
                                                                                                       fair_error)
                effect_distance_mean = np.mean([effect_distance_pos, effect_distance_neg])
                effect_distance_contrained_mean = np.mean(
                    [effect_distance_contrained_pos, effect_distance_contrained_neg])

                # print('checking effect_distance_pos error end..')

                kl_mean = np.mean([kl_pos, kl_neg])
                js_mean = np.mean([js_pos, js_neg])
                hellinger_mean = np.mean([hellinger_pos, hellinger_neg])
                wasserstein_mean = np.mean([wasserstein_distance_pos, wasserstein_distance_neg])
                # cramers_v_mean = np.mean([cramers_v_pos, cramers_v_neg])
                total_variation_mean = np.mean([total_variation_distance_pos, total_variation_distance_neg])
                # print('checking mean 1 end..')
            elif len(a_before_sub) > 1:
                # print('checking mean 2 starts..')
                kl_pos, js_pos = DistanceMeasure2.js_divergence(a_before_sub_indirect, a_after_sub_indirect)
                hellinger_pos = DistanceMeasure2.hellinger_continous_1d(a_before_sub_indirect, a_after_sub_indirect)
                wasserstein_distance_pos = DistanceMeasure2.wasserstein_distance(a_before_sub_indirect, a_after_sub_indirect)
                # cramers_v_pos = DistanceMeasure.cramers_v(a_before_sub, a_after_sub)

                total_variation_distance_pos = DistanceMeasure2.total_variation_distance(a_before_sub_indirect, a_after_sub_indirect)
                effect_distance_pos, effect_distance_contrained_pos = DistanceMeasure2.effect_distance(a_before_sub_indirect, a_after_sub_indirect,
                                                                                                       fair_error)

                effect_distance_mean = effect_distance_pos
                effect_distance_contrained_mean = effect_distance_contrained_pos
                kl_mean = kl_pos
                js_mean = js_pos
                hellinger_mean = hellinger_pos
                wasserstein_mean = wasserstein_distance_pos
                # cramers_v_mean = cramers_v_pos
                total_variation_mean = total_variation_distance_pos
                # print('checking mean 2 ends..')
            elif len(b_before_sub) > 1:
                # print('checking mean 3 starts..')
                kl_neg, js_neg = DistanceMeasure2.js_divergence(b_before_sub_indirect, b_after_sub_indirect)
                hellinger_neg = DistanceMeasure2.hellinger_continous_1d(b_before_sub_indirect, b_after_sub_indirect)
                wasserstein_distance_neg = DistanceMeasure2.wasserstein_distance(b_before_sub_indirect, b_after_sub_indirect)
                # cramers_v_neg = DistanceMeasure.cramers_v(b_before_sub, b_after_sub)
                total_variation_distance_neg = DistanceMeasure2.total_variation_distance(b_before_sub_indirect, b_after_sub_indirect)
                # import matplotlib.pyplot as plt
                # fig, ax = plt.subplots(1, 1)
                effect_distance_neg, effect_distance_contrained_neg = DistanceMeasure2.effect_distance(b_before_sub_indirect, b_after_sub_indirect,
                                                                                                       fair_error)

                effect_distance_mean = effect_distance_neg
                effect_distance_contrained_mean = effect_distance_contrained_neg

                kl_mean = kl_neg
                js_mean = js_neg
                hellinger_mean = hellinger_neg
                wasserstein_mean = wasserstein_distance_neg
                # cramers_v_mean = cramers_v_neg
                total_variation_mean = total_variation_distance_neg
                # print('checking mean 3 ends..')
            # print('checking cramers_v_mean error ends..')
            # print('checking 1 starts..')
            cramers_v_mean = DistanceMeasure.cramers_v(category_target_predicted_before_indirect,
                                                       category_target_predicted_after_indirect)

            # print('checking 1 ends..')

            # print('total_variation_mean and cramers_v_mean: ', total_variation_mean,cramers_v_mean, category_target_predicted_before, category_target_predicted_after)
            # print('checking 2 starts..')
            if not category_1 in data_category.keys():
                data_category[category_1] = {}

                # if category_2 in data_category[category_1].keys():
            if not category_2 in data_category[category_1].keys():
                data_category[category_1][category_2] = {}
            data_category[category_1][category_2][IMPACT_TYPE] = {}
            data_category[category_1][category_2][IMPACT_TYPE]['ACC'] = 0
            data_category[category_1][category_2][IMPACT_TYPE]['TPR'] = 0
            data_category[category_1][category_2][IMPACT_TYPE]['FPR'] = 0
            data_category[category_1][category_2][IMPACT_TYPE]['distortion_multiple'] = distortion_multiple
            data_category[category_1][category_2][IMPACT_TYPE]['distortion_feature'] = distortion_feature

            data_category[category_1][category_2][IMPACT_TYPE]['SP'] = 0
            data_category[category_1][category_2][IMPACT_TYPE]['ACC_after'] = 0
            data_category[category_1][category_2][IMPACT_TYPE]['TPR_after'] = 0
            data_category[category_1][category_2][IMPACT_TYPE]['FPR_after'] = 0
            data_category[category_1][category_2][IMPACT_TYPE]['SP_after'] = 0

            data_category[category_1][category_2][IMPACT_TYPE]['proportion'] = round(
                proportion * 100 / len(category_target_original_indirect), 3)
            data_category[category_1][category_2][IMPACT_TYPE]['Importance'] = round(
                f_importance * 100 / len(category_target_original_indirect), 3)
            data_category[category_1][category_2][IMPACT_TYPE]['hellinger_div'] = hellinger_mean
            data_category[category_1][category_2][IMPACT_TYPE]['wasserstein_distance_div'] = wasserstein_mean
            data_category[category_1][category_2][IMPACT_TYPE]['cramers_v_div'] = cramers_v_mean
            data_category[category_1][category_2][IMPACT_TYPE]['total_variation_div'] = total_variation_mean
            data_category[category_1][category_2][IMPACT_TYPE]['effect_distance'] = effect_distance_mean
            data_category[category_1][category_2][IMPACT_TYPE]['effect_distance_contrained'] = effect_distance_contrained_mean
            data_category[category_1][category_2][IMPACT_TYPE]['KL_Divergence'] = kl_mean
            data_category[category_1][category_2][IMPACT_TYPE]['JS_Divergence'] = js_mean

        #print('checking 2 ends..')

        return data_category

    def compute_metrics_indirect(self, perc_parity, x_test_altered, random_index_, category_indices, y_predicted_before, y_predicted_after, column_id,column_id2, p_pred,
                                  q_pred,distortion_multiple,distortion_feature, runkl=False):
        # for column_id in range(self.x_train.shape[1]):
        #    if is_categorical(self.x_test,column_id):
        p_pred = abs(p_pred)
        q_pred = abs(q_pred)
        data_category = {}
        if is_categorical(self.x_test, column_id):
            # todo: first feature is discrete
            category_list = self.get_categorical_features(column_id)
            for cat_ in category_list:
                val = [cat_]
                category_1 = str(val[0])
                if is_categorical(self.x_test, column_id2):
                    # todo: second feature is discrete
                    data_category = {}
                    category_list2 = self.get_categorical_features(column_id2)
                    for cat2_ in category_list2:
                        val2 = [cat2_]
                        category_2 = str(val2[0])
                        ## todo: compute the metrics here
                        #print('compute metrics_1')
                        self.compute_(data_category, perc_parity, x_test_altered, random_index_, category_indices,
                                      y_predicted_before, y_predicted_after, p_pred, q_pred, column_id, column_id2, val,
                                      val2, category_1, category_2, distortion_multiple,distortion_feature)
                        #print('compute metrics_1 end')

                else:
                    category_data2 = self.determine_range(column_id2)
                    #min_val_2 = np.min(list(category_data2.values()))
                    list_val = []
                    for k, v in category_data2.items():
                        list_val.extend(v)
                    # va_list = [vv for vv in list(category_data.values())]
                    min_val_2 = np.min(list_val)

                    for key2, val2 in category_data2.items():

                        # todo: second feature is continous
                        if len(val2) == 1:
                            category_2 = str(val2[0])
                        else:
                            # max_val = np.max(list(category_data.values()))
                            if np.min(val2) == min_val_2:
                                category_2 = '<=' + str(np.max(val2))
                            else:
                                category_2 = '>=' + str(np.min(val2))
                        ## todo: compute the metrics here
                        #print('compute metrics_2')
                        self.compute_(data_category, perc_parity, x_test_altered, random_index_, category_indices,
                                      y_predicted_before, y_predicted_after, p_pred, q_pred, column_id, column_id2, val,
                                      val2, category_1, category_2, distortion_multiple,distortion_feature)
                        #print('compute metrics_2 ends')

        else:
            category_data = self.determine_range(column_id)
            #min_val_1 = np.min(list(category_data.values()))
            # max_val = np.max(list(category_data.values()))
            list_val = []
            for k, v in category_data.items():
                list_val.extend(v)
            # va_list = [vv for vv in list(category_data.values())]
            min_val_1 = np.min(list_val)

            for key, val in category_data.items():
                #todo: first feature is continous
                if len(val) == 1:
                    category_1 = str(val[0])
                else:
                    #max_val = np.max(list(category_data.values()))
                    if np.min(val) == min_val_1:
                        category_1 = '<=' + str(np.max(val))
                    else:
                        category_1 = '>=' + str(np.min(val))

                if is_categorical(self.x_test, column_id2):
                    # todo: second feature is discrete
                    data_category = {}
                    category_list2 = self.get_categorical_features(column_id2)
                    for cat2_ in category_list2:
                        val2 = [cat2_]
                        category_2 = str(val2[0])
                        ## todo: compute the metrics here
                        #print('compute metrics_3')
                        self.compute_(data_category, perc_parity, x_test_altered, random_index_, category_indices,
                                      y_predicted_before, y_predicted_after, p_pred, q_pred, column_id, column_id2, val,
                                      val2, category_1, category_2, distortion_multiple,distortion_feature)
                        #print('compute metrics_3 ends')

                else:
                    category_data2 = self.determine_range(column_id2)
                    #min_val_2 = np.min(list(category_data2.values()))
                    list_val = []
                    for k, v in category_data2.items():
                        list_val.extend(v)
                    # va_list = [vv for vv in list(category_data.values())]
                    min_val_2 = np.min(list_val)

                    for key2, val2 in category_data2.items():

                        # todo: second feature is continous
                        if len(val2) == 1:
                            category_2 = str(val2[0])
                        else:
                            # max_val = np.max(list(category_data.values()))
                            if np.min(val2) == min_val_2:
                                category_2 = '<=' + str(np.max(val2))
                            else:
                                category_2 = '>=' + str(np.min(val2))
                        ## todo: compute the metrics here
                        #print('compute metrics_4')
                        self.compute_(data_category, perc_parity,x_test_altered,random_index_,category_indices,y_predicted_before,y_predicted_after,p_pred, q_pred,column_id, column_id2, val, val2, category_1, category_2, distortion_multiple,distortion_feature)
                        #print('compute metrics_4 ends')
        return data_category

    def fit_nn(self, iterations=10, swap_proportion=[0.1, 0.3, 0.5, 0.7], columns_names=[]):
        y_train = to_categorical(self.y_train)
        y_test = to_categorical(self.y_test)
        n_classes = y_test.shape[1]
        # NNmodel = NNClassifier(self.x_train.shape[1], n_classes)
        # NNmodel.fit(self.x_train, y_train) 0.01,
        #NNmodel = xgboost.XGBRegressor().fit(self.x_train, y_train)

        X, Y = split_features_target(self.data, index=self.target_index)
        #kf = model_selection.KFold(n_splits=10)
        #kf = model_selection.StratifiedKFold(n_splits=10)
        kf = model_selection.KFold(n_splits=10)
        # fill the new kfold column
        feature_list = ['race', 'sex', 'age', 'hours-per-week', 'capital-gain', 'capital-loss']
        feature_list = ['sex', 'age', 'thalach', 'ca', 'thal', 'exang', 'cp', 'trestbps', 'restecg', 'fbs', 'oldpeak',
                        'chol']
        feature_list = ['sex', 'age', 'health', 'Pstatus', 'nursery', 'Medu', 'Fjob', 'schoolsup', 'absences',
                        'activities', 'higher', 'traveltime', 'paid', 'guardian', 'Walc', 'freetime', 'famsup',
                        'romantic', 'studytime', 'goout', 'reason', 'famrel', 'internet']
        feature_list = ['Sex', 'Age', 'Job', 'Saving', 'Checking', 'Credit', 'Housing', 'Purpose']
        #feature_list = ['race', 'sex', 'age', 'c_charge_degree', 'priors_count']
        #feature_list = ['age', 'education', 'job', 'loan',  'balance', 'housing',  'duration', 'campaign','default']


        feature_list = ['sex', 'age', 'health', 'Pstatus', 'nursery', 'Medu', 'Fjob', 'schoolsup', 'absences',
                        'activities', 'higher', 'traveltime', 'paid', 'guardian', 'Walc', 'freetime', 'famsup',
                        'romantic', 'studytime', 'goout', 'reason', 'famrel', 'internet']
        feature_list = ['age', 'education', 'job', 'loan', 'balance', 'housing', 'duration', 'campaign', 'default']
        #feature_list = ['sex', 'age', 'thalach', 'ca', 'thal', 'exang', 'cp', 'trestbps', 'restecg', 'fbs', 'oldpeak',
        #                'chol']

        #print(Y)

        for fold, (train_idx, test_idx) in enumerate(kf.split(X=self.df_data,y=Y)):
            #for fold, (train_idx, test_idx) in enumerate(kf.split(X=self.df_data, y=np.array(Y).reshape(-1,1))):
            train, test = self.df_data.loc[train_idx], self.df_data.loc[test_idx]
            train, test = train.to_numpy(), test.to_numpy()
            print(fold, 'train: ', train.shape, test.shape)
            self.x_train, self.y_train = split_features_target(train, self.target_index)
            self.x_test, self.y_test = split_features_target(test, self.target_index)
            y_train = to_categorical(self.y_train)
            y_test = to_categorical(self.y_test)
            NNmodel = Models2(self.x_train, self.x_test, y_train, y_test).decision_tree_regressor()
            explainer = shap.TreeExplainer(NNmodel)
            p_pred = NNmodel.predict(self.x_test)
            a_before, b_before = [], []
            list_before, list_after = [], []
            for pred_ in p_pred:
                #print('pred_: ', pred_)
                # print('predicted: ', np.around(pred_))
                a_before.append(pred_[0])
                b_before.append(pred_[1])

                arg_max = pred_.argmax(axis=-1)
                list_before.append(arg_max)

            #N = round(len(self.x_test) / iterations)
            y_sampled_k = self.y_test
            category_indices = []
            for column_id in range(self.x_train.shape[1]):
                if is_categorical(self.x_test, column_id):
                    category_indices.append(column_id)

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

                random_index_1 = [i for i in range(len(self.y_test))]
                # print('random_index: ', random_index, swap_proportion)
                #print(len(random_index_1))
                #print(len(random_index_1) * swap_proportion)

                if round(len(random_index_1) * swap_proportion_, 0) == len(random_index_1):  # and swap_proportion > 0.9:
                    _, random_index_ = 0, random_index_1
                else:
                    _, random_index_ = data_split(data=random_index_1, sample_size=swap_proportion_)
                x_sampled_n = []
                for i in range(len(self.x_test)):
                    if i in random_index_:
                        x_sampled_n.append(self.x_test[i])
                shap_values = explainer.shap_values(np.array(x_sampled_n))
                vals = np.abs(shap_values).mean(0)
                for f_name, val, f_import in zip(columns_names, sum(vals), NNmodel.feature_importances_):
                    # print(f_name, val, f_import)
                    self.data_writer3.writerow([fold, swap_proportion_, f_name, val, f_import])
                for column_id in range(self.x_train.shape[1]):
                    feat1 = colums_list[column_id]
                    x_sampled_k_feature = self.get_features_by_index(self.x_test, column_id)
                    a_after, b_after = [], []
                    if is_categorical(self.x_test, column_id):
                        random_index_, x_test_altered = alter_feature_values_categorical_3(self.x_test, self.y_test,
                                                                                           self.get_categorical_features(
                                                                                               column_id),
                                                                                           column_id,
                                                                                           swap_proportion=swap_proportion_)
                        # q_pred = NNmodel.predict(tf.convert_to_tensor(x_test_altered, dtype=tf.float32))
                        q_pred = NNmodel.predict(x_test_altered)

                        # print('q_pred: ', q_pred)
                        list_after = []
                        for pred_ in q_pred:
                            # pred_ = abs(pred_)
                            a_after.append(pred_[0])
                            b_after.append(pred_[1])
                            arg_max = pred_.argmax(axis=-1)
                            # print('q_pred: ', pred_, pred_2, abs(pred_), abs(pred_[0]), abs(pred_[1]), arg_max)
                            list_after.append(arg_max)

                        x_test_altered_feature = self.get_features_by_index(x_test_altered, column_id)

                        distortion_multiple = DistanceMeasure2._distance_multiple(self.x_test, x_test_altered,
                                                                                  category_indices, alpha=alpha)
                        distortion_feature = DistanceMeasure2._hamming_distance(x_sampled_k_feature,
                                                                                x_test_altered_feature)

                        #print('Check category single stars here ----')
                        data_category_after = self.compute_metrics_direct(alpha, x_test_altered, random_index_,
                                                                   category_indices, list_before, list_after, column_id,
                                                                   p_pred,
                                                                   q_pred,distortion_multiple, distortion_feature, runkl=True)
                        #print('Check category single ends here ----')
                    else:
                        random_index_, x_test_altered = alter_feature_value_continous_3(self.x_test, self.y_test,
                                                                                        self.determine_range(column_id),
                                                                                        column_id,
                                                                                        swap_proportion=swap_proportion_)
                        # q_pred = NNmodel.predict(tf.convert_to_tensor(x_test_altered, dtype=tf.float32))
                        q_pred = NNmodel.predict(x_test_altered)
                        list_after = []
                        for pred_ in q_pred:
                            # pred_ = abs(pred_)
                            a_after.append(pred_[0])
                            b_after.append(pred_[1])
                            arg_max = pred_.argmax(axis=-1)
                            list_after.append(arg_max)
                        x_sampled_k_feature = self.get_features_by_index(self.x_test, column_id)
                        x_test_altered_feature = self.get_features_by_index(x_test_altered, column_id)

                        distortion_multiple = DistanceMeasure2._distance_multiple(self.x_test, x_test_altered,
                                                                                  category_indices, alpha=alpha)
                        distortion_feature = DistanceMeasure2._quared_error_proportion(x_sampled_k_feature,
                                                                                       x_test_altered_feature)


                        data_category_after = self.compute_metrics_continous_direct(alpha, x_test_altered, random_index_,
                                                                             category_indices, list_before, list_after,
                                                                             column_id, p_pred, q_pred,distortion_multiple, distortion_feature, runkl=True)
                        #print('Check continous single ends here ----')
                    for key,val in data_category_after.items():
                        self.data_writer2.writerow(
                            ['F_{}'.format(column_id), 'F_{}'.format(column_id), self.colums_list[column_id],
                             self.colums_list[column_id], key, key, swap_proportion_, 'CDI',
                             data_category_after[key]['ACC'],
                             data_category_after[key]['ACC_after'],
                             data_category_after[key]['distortion_multiple'],
                             data_category_after[key]['distortion_feature'],
                             data_category_after[key]['hellinger_div'],
                             data_category_after[key]['wasserstein_distance_div'],
                             data_category_after[key]['cramers_v_div'],
                             data_category_after[key]['total_variation_div'],
                             data_category_after[key]['KL_Divergence'],
                             data_category_after[key]['JS_Divergence'],
                             data_category_after[key]['proportion'],
                             data_category_after[key]['Importance'],
                             data_category_after[key]['effect_distance'],
                             data_category_after[key]['effect_distance_contrained'], data_category_after[key]['SP'],
                             data_category_after[key]['SP_after'], data_category_after[key]['TPR'],
                             data_category_after[key]['TPR_after'], data_category_after[key]['FPR'],
                             data_category_after[key]['FPR_after']])
                    for column_id2 in range(self.x_train.shape[1]):
                        feat2 = colums_list[column_id2]
                        if column_id != column_id2 and feature_list.index(feat1) <= feature_list.index(feat2):
                            _, x_test_altered = alter_2_columns(self.x_test, self.y_test, self.get_posible_values_2_columns([column_id, column_id2]),
                                                                                     feature_indices=[column_id, column_id2],
                                                                                     swap_proportion=swap_proportion_)
                            x_sampled_k_feature = self.get_features_by_index(self.x_test, [column_id, column_id2])
                            # if column_id != self.sensitive_index:
                            # print('Column type name: ',self.data[0:, column_id].dtype.name)
                            a_after, b_after = [], []
                            #if is_categorical(self.x_test, column_id):
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


                            x_test_altered_feature = self.get_features_by_index(x_test_altered, [column_id,column_id2])


                            #distortion_multiple = DistanceMeasure2._distance_multiple(self.x_test, x_test_altered,
                            #                                                          category_indices, alpha=alpha)
                            #distortion_feature = DistanceMeasure2._quared_error_proportion(x_sampled_k_feature,
                            #                                                               x_test_altered_feature)


                            #kl_div_distortion, js_div_distortion = DistanceMeasure2.js_divergence(x_sampled_k_feature,
                            #                                                                         x_test_altered_feature)
                            #hellinger_distortion = DistanceMeasure2.hellinger_continous_1d(x_sampled_k_feature,
                            #                                                                         x_test_altered_feature)
                            #wasserstein_distance_distortion =  DistanceMeasure2.wasserstein_distance(x_sampled_k_feature,
                            #                                                                         x_test_altered_feature)
                            #cramers_v_distortion = DistanceMeasure.cramers_v(x_sampled_k_feature,x_test_altered_feature)
                            #total_variation_distortion = DistanceMeasure2.total_variation_distance(x_sampled_k_feature,
                            #                                                                         x_test_altered_feature)

                            #print('sampled k feature distortion values: ', kl_div_distortion, hellinger_distortion, wasserstein_distance_distortion, cramers_v_distortion, total_variation_distortion)

                            #kl_div_distortion, js_div_distortion = DistanceMeasure2.js_divergence_2d(x_sampled_k, x_test_altered)

                            #wasserstein_distance_distortion = DistanceMeasure2.wasserstein_distance_pdf2d(x_sampled_k, x_test_altered)

                            #hellinger_distortion = DistanceMeasure.hellinger_multivariate(self.x_test, x_test_altered)

                            #cramers_v_distortion = DistanceMeasure.cramers_v(x_sampled_k, x_test_altered)

                            distortion_multiple, distortion_feature = 0,0 #hellinger_distortion, hellinger_distortion

                            #total_variation_distortion = DistanceMeasure2.total_variation_distance_2d(x_sampled_k, x_test_altered)
                            #print('cramers_v: ', cramers_v)

                            #perc_parity, x_test_altered, random_index_, category_indices, y_predicted_before, y_predicted_after, column_id, column_id2, p_pred,
                            #q_pred, distortion_multiple, distortion_feature

                            #print('Check compute_metrics_indirect function starts here ----')
                            data_category_2_columns_after = self.compute_metrics_indirect(alpha, x_test_altered,
                                                                                                  random_index_,
                                                                                                  category_indices,
                                                                                                  list_before,
                                                                                                  list_after,
                                                                                                  column_id,column_id2, p_pred,
                                                                                                  q_pred,
                                                                                                  distortion_multiple,
                                                                                                  distortion_feature,
                                                                                                  runkl=True)
                            #print('Check compute_metrics_indirect function ends here ----')

                            for key, val in data_category_2_columns_after.items():
                                for key2_, val2_ in val.items():
                                    for key2, val2 in val2_.items():
                                        '''
                                        self.data_writer.writerow(
                                        ['F_{}'.format(column_id),'F_{}'.format(column_id2), self.colums_list[column_id],self.colums_list[column_id2],swap_proportion_, acc_category, acc_category_after,
                                         distortion_multiple, distortion_feature, #cramers_v_distortion, total_variation_distortion, kl_div_distortion, js_div_distortion,
                                         hellinger_mean, wasserstein_mean, cramers_v_mean, total_variation_mean,
                                         kl_mean, js_mean, casuality, f_importance,  SP, SP_after, TPR, TPR_after, FPR,
                                         FPR_after])
                                        '''
                                        self.data_writer2.writerow(
                                            ['F_{}'.format(column_id), 'F_{}'.format(column_id2), self.colums_list[column_id],self.colums_list[column_id2], key, key2_, swap_proportion_, key2,
                                             val2['ACC'],
                                             val2['ACC_after'],
                                             val2['distortion_multiple'],
                                             val2['distortion_feature'],
                                             val2['hellinger_div'],
                                             val2['wasserstein_distance_div'],
                                             val2['cramers_v_div'],
                                             val2['total_variation_div'],
                                             val2['KL_Divergence'],
                                             val2['JS_Divergence'],
                                             val2['proportion'],
                                             val2['Importance'],
                                             val2['effect_distance'],
                                             val2['effect_distance_contrained'], val2['SP'],
                                             val2['SP_after'], val2['TPR'],
                                             val2['TPR_after'], val2['FPR'],
                                             val2['FPR_after']])

                            kl_mean, js_mean, hellinger_mean, wasserstein_mean, cramers_v_mean, total_variation_mean, effect_distance_mean,effect_distance_contrained_mean  = 0, 0, 0, 0, 0, 0, 0, 0
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

                                effect_distance_pos, effect_distance_contrained_pos = DistanceMeasure2.effect_distance(
                                    a_before, a_after, fair_error)
                                effect_distance_neg, effect_distance_contrained_neg = DistanceMeasure2.effect_distance(
                                    b_before, b_after, fair_error)
                                effect_distance_mean = np.mean([effect_distance_pos, effect_distance_neg])
                                effect_distance_contrained_mean = np.mean(
                                    [effect_distance_contrained_pos, effect_distance_contrained_neg])


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
                                effect_distance_pos, effect_distance_contrained_pos = DistanceMeasure2.effect_distance(
                                    a_before, a_after, fair_error)

                                kl_mean = kl_pos
                                js_mean = js_pos
                                hellinger_mean = hellinger_pos
                                wasserstein_mean = wasserstein_distance_pos
                                cramers_v_mean = cramers_v_pos
                                total_variation_mean = total_variation_distance_pos

                                effect_distance_mean = effect_distance_pos
                                effect_distance_contrained_mean = effect_distance_contrained_pos
                            elif len(b_before) > 1:
                                kl_neg, js_neg = DistanceMeasure2.js_divergence(b_before, b_after)
                                hellinger_neg = DistanceMeasure2.hellinger_continous_1d(b_before, b_after)
                                wasserstein_distance_neg = DistanceMeasure2.wasserstein_distance(b_before, b_after)
                                cramers_v_neg = DistanceMeasure.cramers_v(b_before, b_after)
                                total_variation_distance_neg = DistanceMeasure2.total_variation_distance(b_before, b_after)
                                effect_distance_neg, effect_distance_contrained_neg = DistanceMeasure2.effect_distance(
                                    b_before, b_after, fair_error)

                                kl_mean = kl_neg
                                js_mean = js_neg
                                hellinger_mean = hellinger_neg
                                wasserstein_mean = wasserstein_distance_neg
                                cramers_v_mean = cramers_v_neg
                                total_variation_mean = total_variation_distance_neg

                                effect_distance_mean = effect_distance_neg
                                effect_distance_contrained_mean = effect_distance_contrained_neg
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
                                    ['F_{}'.format(column_id),'F_{}'.format(column_id2), self.colums_list[column_id],self.colums_list[column_id2],swap_proportion_,'NII', acc_category, acc_category_after,
                                     distortion_multiple, distortion_feature, #cramers_v_distortion, total_variation_distortion, kl_div_distortion, js_div_distortion,
                                     hellinger_mean, wasserstein_mean, cramers_v_mean, total_variation_mean,
                                     kl_mean, js_mean, casuality, f_importance, effect_distance_mean, effect_distance_contrained_mean,  SP, SP_after, TPR, TPR_after, FPR,
                                     FPR_after])
        if self.log_data:
            self.data_file.close()
            self.data_file_2.close()
            self.data_file_3.close()

if __name__ == '__main__':
    path = '../dataset/'
    ### Adult dataset
    # target_column = 'Probability'
    fair_error = 0.01
    alpha = 0.3
    correlation_threshold = 0.45  # 0.35
    loadData = LoadData(path, threshold=correlation_threshold)  # ,threshold=correlation_threshold
    data_name = 'bank-{}_'.format(alpha)  # _35_threshold

    #df_adult = loadData.load_adult_data('adult.data.csv')
    # df_adult = loadData.load_credit_defaulter('default_of_credit_card_clients.csv')
    #df_adult = loadData.load_clevelan_heart_data('processed.cleveland.data.csv')
    #df_adult = loadData.load_student_data('Student.csv')
    #df_adult = loadData.load_german_data2('german_credit_data.csv')
    #df_adult = loadData.load_german_data('GermanData.csv')
    #df_adult = loadData.load_compas_data('compas-scores-two-years.csv')
    df_adult = loadData.load_bank_data('bank.csv')
    sensitive_list = loadData.sensitive_list
    sensitive_indices = loadData.sensitive_indices
    colums_list = df_adult.columns.tolist()

    #data_name = 'clevelan_heart'
    #data_name = 'adult'
    #data_name = 'Student'
    #path = '../dataset/experiments/data/'
    #load_data = LoadTrainTest(path)
    #train, test, target_index = load_data.load_adult(data_name)

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

    print(target_name, colums_list, df_adult)
    for colum in colums_list:
        print(set(df_adult[target_name].values.tolist()))
#        print(colum, df_adult[target_name].corr(df_adult[colum]))
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
    sensitivityAnalysis.fit_nn(columns_names=colums_list[:-1])

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
