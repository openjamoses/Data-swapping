import csv
import os
import random

import xgboost
from tensorflow.python.keras.utils.np_utils import to_categorical

from src.models.data.kl_jn_divergences import kl_js_divergence_metrics
from src.models.data.load_train_test import LoadTrainTest
from src.models.v3.NNClassifier import NNClassifier
from src.models.v3.load_data import LoadData
from src.models.v3.ranking import Ranking
from src.models.v3.sensitivity_utils import *
from src.models.v3.utility_functions import *
import seaborn as sn
import matplotlib.pyplot as plt
import tensorflow as tf


class SensitivityAnalysis:
    def __init__(self, data, target_index=None, sensitive_name='sex', sensitive_index=8, log_data=True,
                 log_path='../dataset/logging2/', data_name='', colums_list=[], threshold=None):
        self.data = data
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
            ['ID', 'Feature', 'Acc', 'Acc_after', 'KL_Divergence', 'JS_Divergence', 'Casuality', 'Importance', 'SP', 'SP_after',
             'TPR', 'TPR_after', 'FPR', 'FPR_after'])

        self.log_path_local = log_path + 'KL_Divergence_local_{}.csv'.format(data_name)
        self.data_file2 = open(self.log_path_local, mode='w', newline='',
                               encoding='utf-8')
        self.data_writer2 = csv.writer(self.data_file2)
        self.data_writer2.writerow(
            ['ID', 'Feature', 'Category', 'Acc', 'Acc_after', 'KL_Pos', 'JS_Divergence', 'Casuality', 'Importance', 'SP',
             'SP_after', 'TPR', 'TPR_after', 'FPR', 'FPR_after'])

        print('Sensitive attribute: ', sensitive_name)

    def data_init(self, data):
        self.train, self.test = data_split(data=data, sample_size=0.25)
        self.x_train, self.y_train = split_features_target(self.train, index=self.target_index)
        self.x_test, self.y_test = split_features_target(self.test, index=self.target_index)

        print('self.y_test: ', self.y_test)

    def get_categorical_features(self, feature_index):
        print('column data: ', set(self.data[0:, feature_index]))
        return list(set(self.data[0:, feature_index]))

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
                print('Category <3: ', np.min(list(val)), percentile_50, percentile_100)
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

        # folded_data[0] = [np.min(data_range), percentile_25]
        # folded_data[1] = [percentile_25, percentile_50]
        # folded_data[2] = [percentile_50, np.max(data_range)+1]

        # print('Percentiles: ', percentile_25, percentile_50, percentile_75)
        '''for i in range(len(sorted_)):
            if sorted_[i] <= percentile_25:
                fold_count = 0
                if fold_count in folded_data.keys():
                    folded_data[fold_count].append(sorted_[i])
                else:
                    folded_data[fold_count] = [sorted_[i]]
            elif sorted_[i] > percentile_25 and sorted_[i] <= percentile_50:
                fold_count = 1
                if fold_count in folded_data.keys():
                    folded_data[fold_count].append(sorted_[i])
                else:
                    folded_data[fold_count] = [sorted_[i]]
            elif sorted_[i] > percentile_50:
                fold_count = 2
                if fold_count in folded_data.keys():
                    folded_data[fold_count].append(sorted_[i])
                else:
                    folded_data[fold_count] = [percentile_50, sorted_[i]]
            #if i%interval_ == 0 and i > 0:
            #    fold_count += 1
            #if fold_count in folded_data.keys():
            #    folded_data[fold_count].append(sorted_[i])
            #else:
            #    folded_data[fold_count] = [sorted_[i]]
        for key, val in folded_data.items():
            #folded_data[key] = list(range(np.min(val), np.max(val)))
            folded_data[key] = [np.min(val), np.max(val)]'''
        # print(self.colums_list[feature_index], folded_data)
        # return folded_data

    def compute_metrics(self, random_indices, y_predicted_before, y_predicted_after, column_id, p_pred=None,
                        q_pred=None, runkl=False):

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

            for i in range(len(random_indices)):
                if self.x_test[random_indices[i]][column_id] == category:
                    category_target_original.append(self.y_test[i])
                    category_target_predicted_before.append(y_predicted_before[random_indices[i]])
                    category_target_predicted_after.append(y_predicted_after[i])
                    if y_predicted_before[random_indices[i]] == 1 and self.y_test[random_indices[i]] == 1:
                        TP += 1
                    if y_predicted_before[random_indices[i]] == 1 and self.y_test[random_indices[i]] == 0:
                        FP += 1
                    if y_predicted_before[random_indices[i]] == 0 and self.y_test[random_indices[i]] == 0:
                        TN += 1
                    if y_predicted_before[random_indices[i]] == 0 and self.y_test[random_indices[i]] == 1:
                        FN += 1
                    if y_predicted_after[i] == 1 and self.y_test[random_indices[i]] == 1:
                        TP_after += 1
                    if y_predicted_after[i] == 1 and self.y_test[random_indices[i]] == 0:
                        FP_after += 1
                    if y_predicted_after[i] == 0 and self.y_test[random_indices[i]] == 0:
                        TN_after += 1
                    if y_predicted_after[i] == 0 and self.y_test[random_indices[i]] == 1:
                        FN_after += 1

                    if runkl:
                        a_before_sub.append(p_pred[random_indices[i]][0])
                        b_before_sub.append(p_pred[random_indices[i]][1])
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

            proportion = np.sum(
                category_target_predicted_before != category_target_predicted_after)  # calculate_proportion(TP, FP, TN, FN)
            kl_pos = None
            kl_neg = None
            if runkl:
                '''kl_pos, p, q, lin = run_dl_divergence(a_before_sub, a_after_sub, 'Pos_before', 'Pos_after',
                                                      column_index=column_id,
                                                      sub_category=self.colums_list[column_id] + ' ({})_Pos'.format(
                                                          category), path=self.new_path)
                kl_neg, p2, q2, lin2 = run_dl_divergence(b_before_sub, b_after_sub, 'Neg_before', 'Neg_after',
                                                         column_index=column_id,
                                                         sub_category=self.colums_list[column_id] + ' ({})_Neg'.format(
                                                             category), path=self.new_path)'''

                kl_pos, js_pos = kl_js_divergence_metrics(a_before_sub, a_after_sub)
                kl_neg, js_neg = kl_js_divergence_metrics(b_before_sub, b_after_sub)
            kl_mean = np.mean([kl_pos, kl_neg])
            js_mean = np.mean([js_pos, js_neg])
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
            data_category[category]['KL_Divergence'] = kl_mean
            data_category[category]['JS_Divergence'] = js_mean
        return data_category

    def compute_metrics_continous(self, random_indices, y_predicted_before, y_predicted_after, column_id, p_pred=None,
                                  q_pred=None, runkl=False):
        # for column_id in range(self.x_train.shape[1]):
        #    if is_categorical(self.x_test,column_id):
        p_pred = abs(p_pred)
        q_pred = abs(q_pred)
        data_category = {}
        category_data = self.determine_range(column_id)

        print('category_data: ', category_data)
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
            for i in range(len(random_indices)):
                if self.x_test[random_indices[i]][column_id] in val:
                    category_target_original.append(self.y_test[random_indices[i]])
                    category_target_predicted_before.append(y_predicted_before[random_indices[i]])
                    category_target_predicted_after.append(y_predicted_after[i])
                    # print(y_predicted[i], len(self.y_test[i]), self.y_test[i], self.y_test)
                    if y_predicted_before[random_indices[i]] == 1 and self.y_test[random_indices[i]] == 1:
                        TP += 1
                    if y_predicted_before[random_indices[i]] == 1 and self.y_test[random_indices[i]] == 0:
                        FP += 1
                    if y_predicted_before[random_indices[i]] == 0 and self.y_test[random_indices[i]] == 0:
                        TN += 1
                    if y_predicted_before[random_indices[i]] == 0 and self.y_test[random_indices[i]] == 1:
                        FN += 1
                    if y_predicted_after[i] == 1 and self.y_test[random_indices[i]] == 1:
                        TP_after += 1
                    if y_predicted_after[i] == 1 and self.y_test[random_indices[i]] == 0:
                        FP_after += 1
                    if y_predicted_after[i] == 0 and self.y_test[random_indices[i]] == 0:
                        TN_after += 1
                    if y_predicted_after[i] == 0 and self.y_test[random_indices[i]] == 1:
                        FN_after += 1
                    if runkl:
                        a_before_sub.append(p_pred[random_indices[i]][0])
                        b_before_sub.append(p_pred[random_indices[i]][1])
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

            proportion = np.sum(
                category_target_predicted_before != category_target_predicted_after)  # calculate_proportion(TP, FP, TN, FN)
            kl_pos = None
            kl_neg = None
            if len(val) > 0:
                if len(val) == 1:
                    category = str(val[0])
                else:
                    category = str(np.min(val)) + '-' + str(np.max(val))

                if runkl:
                    '''kl_pos, p, q, lin = run_dl_divergence(a_before_sub, a_after_sub, 'Pos_before', 'Pos_after',
                                                          column_index=column_id,
                                                          sub_category=self.colums_list[column_id] + ' ({})_Pos'.format(
                                                              category), path=self.new_path)
                    # print('kl test: ', len(b_before_sub), len(b_after_sub), b_before_sub, b_after_sub)
                    kl_neg, p2, q2, lin2 = run_dl_divergence(b_before_sub, b_after_sub, 'Neg_before', 'Neg_after',
                                                             column_index=column_id, sub_category=self.colums_list[
                                                                                                      column_id] + ' ({})_Neg'.format(
                            category), path=self.new_path)'''

                    kl_pos, js_pos = kl_js_divergence_metrics(a_before_sub, a_after_sub)
                    # print('kl test: ', len(b_before_sub), len(b_after_sub), b_before_sub, b_after_sub)
                    kl_neg, js_neg = kl_js_divergence_metrics(b_before_sub, b_after_sub)
                kl_mean = np.mean([kl_pos, kl_neg])
                js_mean = np.mean([js_pos, js_neg])
                data_category[category] = {}
                data_category[category]['ACC'] = round(acc_category * 100 / len(category_target_original), 3)
                data_category[category]['TPR'] = TPR
                data_category[category]['FPR'] = FPR
                data_category[category]['SP'] = SP
                data_category[category]['ACC_after'] = round(acc_category_after * 100 / len(category_target_original),
                                                             3)
                data_category[category]['TPR_after'] = TPR_after
                data_category[category]['FPR_after'] = FPR_after
                data_category[category]['SP_after'] = SP_after
                data_category[category]['proportion'] = round(proportion * 100 / len(category_target_original), 3)
                data_category[category]['Importance'] = round(f_importance * 100 / len(category_target_original), 3)
                data_category[category]['KL_Divergence'] = kl_mean
                data_category[category]['JS_Divergence'] = js_mean
        return data_category

    def fit_nn(self, iterations=10):
        y_train = to_categorical(self.y_train)
        y_test = to_categorical(self.y_test)
        n_classes = y_test.shape[1]
        # NNmodel = NNClassifier(self.x_train.shape[1], n_classes)
        # NNmodel.fit(self.x_train, y_train)
        NNmodel = xgboost.XGBRegressor().fit(self.x_train, y_train)

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
        for iter in range(iterations):
            random_index = random.choices(range(len(self.x_test)), k=N)
            x_sampled_k = []
            y_sampled_k = []
            for i in range(len(self.x_test)):
                x_sampled_k.append(self.x_test[i])
                y_sampled_k.append(self.y_test[i])

            # for column_id in range(self.x_train.shape[1]):
            #    if is_categorical(self.x_test,column_id):
            #        data_category = self.compute_metrics(list_before,column_id)
            #    else:
            #        data_category = self.compute_metrics_continous(list_before, column_id)
            #    for key, val in data_category.items():
            #        self.data_writer2.writerow(
            #            ['F_{}'.format(column_id), self.colums_list[column_id],key,'Before', data_category[key]['ACC'], data_category[key]['KL_Pos'], data_category[key]['KL_Neg'], data_category[key]['proportion'], data_category[key]['SP'], data_category[key]['TPR'], data_category[key]['FPR']])

            '''x_test_altered = alter_feature_values_categorical(self.x_test, self.posible_sensitive_values, self.sensitive_index)
            #q_pred = NNmodel.predict(tf.convert_to_tensor(x_test_altered, dtype=tf.float32))
            q_pred = NNmodel.predict(x_test_altered)

            a_after, b_after = [], []

            for pred_ in q_pred:
                a_after.append(pred_[0])
                b_after.append(pred_[1])
                arg_max = pred_.argmax(axis=-1)
                list_before.append(arg_max)

            total_equal_pos = np.sum(a_before == a_after)
            total_equal_neg = np.sum(b_before == b_after)


            total_equal_all = np.sum(list_before == list_after)

            ## Run DL Divergence measure
            kl_div_positive, p, q, lin = run_dl_divergence(a_before, a_after, 'Pos_before', 'Pos_after',column_index=self.sensitive_index)
            kl_div_negative, p2, q2, lin2 = run_dl_divergence(b_before, b_after, 'Neg_before', 'Neg_after', column_index=self.sensitive_index)

            print('KL Divergence for positive class: ', kl_div_positive)
            print('KL Divergence for negative class: ', kl_div_negative)


            if self.log_data:
                self.data_writer2.writerow(['F_{}'.format(self.sensitive_index), 'POS', kl_div_positive, total_equal_pos/len(p_pred), total_equal_all])
                self.data_writer2.writerow(['F_{}'.format(self.sensitive_index), 'NEG', kl_div_negative, total_equal_neg/len(q_pred), total_equal_all])

                for i in range(len(p)):
                    # Positive class
                    self.data_writer.writerow(['F_{}'.format(self.sensitive_index), i, 'POS', 'Before', p[i]])
                    self.data_writer.writerow(['F_{}'.format(self.sensitive_index), i, 'POS', 'After', q[i]])
                # Negative class
                for i in range(len(p2)):
                    self.data_writer.writerow(['F_{}'.format(self.sensitive_index), i, 'NEG', 'Before', p2[i]])
                    self.data_writer.writerow(['F_{}'.format(self.sensitive_index), i, 'NEG', 'After', q2[i]])'''
            for column_id in range(self.x_train.shape[1]):
                # if column_id != self.sensitive_index:
                # print('Column type name: ',self.data[0:, column_id].dtype.name)
                a_after, b_after = [], []
                if is_categorical(self.x_test, column_id):
                    x_test_altered = alter_feature_values_categorical(x_sampled_k,
                                                                      self.get_categorical_features(column_id),
                                                                      column_id)
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
                    data_category_after = self.compute_metrics(random_index, list_before, list_after, column_id, p_pred,
                                                               q_pred, runkl=True)

                else:
                    x_test_altered = alter_feature_value_continous(x_sampled_k, self.determine_range(column_id),
                                                                   column_id)
                    # q_pred = NNmodel.predict(tf.convert_to_tensor(x_test_altered, dtype=tf.float32))
                    q_pred = NNmodel.predict(x_test_altered)
                    list_after = []
                    for pred_ in q_pred:
                        #pred_ = abs(pred_)
                        a_after.append(pred_[0])
                        b_after.append(pred_[1])
                        arg_max = pred_.argmax(axis=-1)
                        list_after.append(arg_max)
                    data_category_after = self.compute_metrics_continous(random_index, list_before, list_after,
                                                                         column_id, p_pred, q_pred, runkl=True)
                for key, val in data_category_after.items():
                    self.data_writer2.writerow(
                        ['F_{}'.format(column_id), self.colums_list[column_id], key, data_category_after[key]['ACC'],
                         data_category_after[key]['ACC_after'], data_category_after[key]['KL_Divergence'],
                         data_category_after[key]['JS_Divergence'], data_category_after[key]['proportion'],
                         data_category_after[key]['Importance'], data_category_after[key]['SP'],
                         data_category_after[key]['SP_after'], data_category_after[key]['TPR'],
                         data_category_after[key]['TPR_after'], data_category_after[key]['FPR'],
                         data_category_after[key]['FPR_after']])

                '''kl_div_positive, p, q, lin = run_dl_divergence(a_before, a_after, 'Pos_before', 'Pos_after',
                                                               column_index=column_id, show_figure=True,
                                                               sub_category=self.colums_list[column_id] + '_Pos',
                                                               path=self.new_path)
                kl_div_negative, p2, q2, lin2 = run_dl_divergence(b_before, b_after, 'Neg_before', 'Neg_after',
                                                                  column_index=column_id, show_figure=True,
                                                                  sub_category=self.colums_list[column_id] + '_Neg',
                                                                  path=self.new_path)'''

                kl_div_positive, js_div_positive = kl_js_divergence_metrics(a_before, a_after)
                kl_div_negative, js_div_negative = kl_js_divergence_metrics(b_before, b_after)

                # self.data_writer2.writerow(
                #    ['F_{}'.format(column_id), self.colums_list[column_id], key, 'Combined', '',
                #     kl_div_positive, kl_div_negative, '',
                #     '', '', ''])

                print('KL Divergence {} for positive class: '.format(column_id), kl_div_positive)
                print('KL Divergence {} for negative class: '.format(column_id), kl_div_negative)

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
                    for i in range(len(random_index)):
                        # print(y_predicted[i], len(self.y_test[i]), self.y_test[i], self.y_test)
                        list_before2.append(list_before[random_index[i]])
                        if list_before[random_index[i]] == 1 and self.y_test[random_index[i]] == 1:
                            TP += 1
                        if list_before[random_index[i]] == 1 and self.y_test[random_index[i]] == 0:
                            FP += 1
                        if list_before[random_index[i]] == 0 and self.y_test[random_index[i]] == 0:
                            TN += 1
                        if list_before[random_index[i]] == 0 and self.y_test[random_index[i]] == 1:
                            FN += 1
                        if list_after[i] == 1 and self.y_test[random_index[i]] == 1:
                            TP_after += 1
                        if list_after[i] == 1 and self.y_test[random_index[i]] == 0:
                            FP_after += 1
                        if list_after[i] == 0 and self.y_test[random_index[i]] == 0:
                            TN_after += 1
                        if list_after[i] == 0 and self.y_test[random_index[i]] == 1:
                            FN_after += 1

                    # category_target_original = np.array(category_target_original)
                    list_before = np.array(list_before)
                    list_before2 = np.array(list_before2)
                    list_after = np.array(list_after)
                    acc_category = round((np.sum(self.y_test == list_before) * 100 / len(self.y_test)), 3)
                    TPR = calculate_TPR(TP, FP, TN, FN)
                    FPR = calculate_FPR(TP, FP, TN, FN)
                    SP = calculate_SP(TP, FP, TN, FN)

                    casuality = round((np.sum(list_before2 != list_after) * 100 / len(self.y_test)), 3)

                    acc_category_after = round((np.sum(y_sampled_k == list_after) * 100 / len(y_sampled_k)), 3)
                    f_importance = round((np.sum(y_sampled_k != list_after) * 100 / len(y_sampled_k)), 3)
                    TPR_after = calculate_TPR(TP_after, FP_after, TN_after, FN_after)
                    FPR_after = calculate_FPR(TP_after, FP_after, TN_after, FN_after)
                    SP_after = calculate_SP(TP_after, FP_after, TN_after, FN_after)

                    self.data_writer.writerow(
                        ['F_{}'.format(column_id), self.colums_list[column_id], acc_category, acc_category_after,
                         np.mean([kl_div_positive, kl_div_negative]), np.mean([js_div_positive, js_div_negative]), casuality, f_importance, SP, SP_after, TPR, TPR_after, FPR,
                         FPR_after])

                    # self.data_writer2.writerow(['F_{}'.format(column_id), self.colums_list[column_id], 'NEG', kl_div_negative, total_equal_neg/len(q_pred), total_equal_all])

                    '''for i in range(len(p)):
                        # Positive class
                        self.data_writer.writerow(['F_{}'.format(column_id),self.colums_list[column_id], i, 'POS', 'Before', p[i], lin[i]])
                        self.data_writer.writerow(['F_{}'.format(column_id),self.colums_list[column_id], i, 'POS', 'After', q[i], lin[i]])
                    # Negative class
                    for i in range(len(p2)):
                        self.data_writer.writerow(
                            ['F_{}'.format(column_id),self.colums_list[column_id], i, 'NEG', 'Before', p2[i], lin2[i]])
                        self.data_writer.writerow(
                            ['F_{}'.format(column_id),self.colums_list[column_id], i, 'NEG', 'After', q2[i], lin2[i]])'''

        if self.log_data:
            self.data_file.close()


if __name__ == '__main__':
    path = '../dataset/'
    ### Adult dataset
    # target_column = 'Probability'
    correlation_threshold = 0.45  # 0.35
    loadData = LoadData(path, threshold=correlation_threshold)  # ,threshold=correlation_threshold
    data_name = 'adult-45_2_50'  # _35_threshold

    df_adult = loadData.load_adult_data('adult.data.csv')
    # df_adult = loadData.load_credit_defaulter('default_of_credit_card_clients.csv')
    #df_adult = loadData.load_student_data('Student.csv')
    # df_adult = loadData.load_student_data('Student.csv')
    sensitive_list = loadData.sensitive_list
    sensitive_indices = loadData.sensitive_indices
    colums_list = df_adult.columns.tolist()

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
    sensitivityAnalysis = SensitivityAnalysis(df_adult.to_numpy(), target_index=target_index,
                                              sensitive_name=sensitive_list[0], sensitive_index=sensitive_indices[0],
                                              data_name=data_name, colums_list=colums_list,
                                              threshold=correlation_threshold)
    sensitivityAnalysis.fit_nn()

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
    data_file_global.close()



