import csv
import os

from tensorflow.python.keras.utils.np_utils import to_categorical

from src.models.v3.NNClassifier import NNClassifier
from src.models.v3.load_data import LoadData
from src.models.v3.sensitivity_utils import *
from src.models.v3.utility_functions import *
import seaborn as sn
import matplotlib.pyplot as plt
import tensorflow as tf

class SensitivityAnalysis:
    def __init__(self, data, target_index=None,sensitive_name='sex', sensitive_index=8, log_data=True, log_path='../dataset/logging/', data_name='', colums_list=[], threshold=None ):
        self.data = data
        self.colums_list = colums_list
        self.sensitive_name = sensitive_name
        self.sensitive_index = sensitive_index
        self.target_index = target_index
        self.posible_sensitive_values = list(np.unique(self.data[0:, self.sensitive_index]))
        self.data_init(data)

        self.log_data = log_data
        self.new_path = log_path
        if self.log_data:
            if threshold == None:
                id = '-baseline'
            else:
                id = threshold
            if not os.path.exists(log_path+str(data_name)):
                os.makedirs(log_path+str(data_name))
            if not os.path.exists(log_path+data_name+"/{}".format(id)):
                os.makedirs(log_path+str(data_name)+ "/{}".format(id))
            self.new_path = log_path+str(data_name)+ "/{}/".format(id)
            self.data_file = open(log_path + 'log_KL_Divergence_overral_{}.csv'.format(data_name), mode='w', newline='', encoding='utf-8')
            self.data_writer = csv.writer(self.data_file)
            self.data_writer.writerow(['ID','Feature','Acc','Acc_after', 'KL_Pos','KL_Neg','Casuality','SP', 'SP_after', 'TPR', 'TPR_after', 'FPR', 'FPR_after'])

            self.data_file2 = open(log_path + 'log_kl_divergence_{}.csv'.format(data_name), mode='w', newline='', encoding='utf-8')
            self.data_writer2 = csv.writer(self.data_file2)
            self.data_writer2.writerow(['ID','Feature','Category', 'Acc','Acc_after', 'KL_Pos', 'KL_Neg','Casuality', 'SP', 'SP_after', 'TPR', 'TPR_after', 'FPR', 'FPR_after'])

        print('Sensitive attribute: ', sensitive_name)
    def data_init(self, data):
        self.train, self.test = data_split(data=data, sample_size=0.25)
        self.x_train, self.y_train = split_features_target(self.train, index=self.target_index)
        self.x_test, self.y_test = split_features_target(self.test, index=self.target_index)

        print('self.y_test: ', self.y_test)
    def get_categorical_features(self, feature_index):
        print('column data: ', set(self.data[0:, feature_index]))
        return list(set(self.data[0:, feature_index]))
    def determine_range(self, feature_index, k_folds=3):
        data_range = self.data[0:, feature_index]
        #interval_ = round(((max(sorted_)-min(sorted_))/k_folds),0)
        fold_count = 0
        folded_data = {}
        percentile_25 = np.percentile(data_range, 25)
        percentile_50 = np.percentile(data_range, 50)
        percentile_75 = np.percentile(data_range, 75)
        percentile_100 = np.percentile(data_range, 100)
        if percentile_50 == percentile_25:
            percentile_50 = np.max(data_range)/2
        if percentile_25 == np.min(data_range):
            percentile_25 = percentile_50/2
        folded_data[0] = [np.min(data_range), percentile_25]
        folded_data[1] = [percentile_25, percentile_50]
        folded_data[2] = [percentile_50, np.max(data_range)+1]

        #print('Percentiles: ', percentile_25, percentile_50, percentile_75)
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
        print(self.colums_list[feature_index], folded_data)
        return folded_data
    def compute_metrics(self, y_predicted_before, y_predicted_after, column_id, p_pred=None, q_pred=None, runkl=False):
        #for column_id in range(self.x_train.shape[1]):
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

                    if runkl:
                        a_before_sub.append(p_pred[i][0])
                        b_before_sub.append(p_pred[i][1])
                        #if q_pred != None:
                        a_after_sub.append(q_pred[i][0])
                        b_after_sub.append(q_pred[i][1])
            category_target_original = np.array(category_target_original)
            category_target_predicted_before = np.array(category_target_predicted_before)
            category_target_predicted_after = np.array(category_target_predicted_after)
            acc_category = np.sum(category_target_original == category_target_predicted_before)
            TPR = calculate_TPR(TP,FP,TN,FN)
            FPR = calculate_FPR(TP, FP, TN, FN)
            SP = calculate_SP(TP, FP, TN, FN)

            acc_category_after = np.sum(category_target_original == category_target_predicted_after)
            TPR_after = calculate_TPR(TP_after, FP_after, TN_after, FN_after)
            FPR_after = calculate_FPR(TP_after, FP_after, TN_after, FN_after)
            SP_after = calculate_SP(TP_after, FP_after, TN_after, FN_after)

            proportion = np.sum(category_target_predicted_before != category_target_predicted_after) #calculate_proportion(TP, FP, TN, FN)
            kl_pos = None
            kl_neg = None
            if runkl:
                kl_pos, p, q, lin = run_dl_divergence(a_before_sub, a_after_sub, 'Pos_before', 'Pos_after',
                                                      column_index=column_id, sub_category=self.colums_list[column_id]+' ({})_Pos'.format(category), path=self.new_path)
                kl_neg, p2, q2, lin2 = run_dl_divergence(b_before_sub, b_after_sub, 'Neg_before', 'Neg_after',
                                                         column_index=column_id, sub_category=self.colums_list[column_id]+' ({})_Neg'.format(category), path=self.new_path)

            data_category[category] = {}
            data_category[category]['ACC'] = round(acc_category*100/len(category_target_original),3)
            data_category[category]['TPR'] = TPR
            data_category[category]['FPR'] = FPR
            data_category[category]['SP'] = SP
            data_category[category]['ACC_after'] = round(acc_category_after * 100 / len(category_target_original), 3)
            data_category[category]['TPR_after'] = TPR_after
            data_category[category]['FPR_after'] = FPR_after
            data_category[category]['SP_after'] = SP_after
            data_category[category]['proportion'] = round(proportion*100/len(category_target_original),3)
            data_category[category]['KL_Pos'] = kl_pos
            data_category[category]['KL_Neg'] = kl_neg
        return data_category

    def compute_metrics_continous(self, y_predicted_before, y_predicted_after,  column_id, p_pred=None, q_pred=None, runkl=False):
        #for column_id in range(self.x_train.shape[1]):
        #    if is_categorical(self.x_test,column_id):
        data_category = {}
        category_data = self.determine_range(column_id)
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
                if self.x_test[i][column_id] >= val[0] and self.x_test[i][column_id] <= val[1]:
                    category_target_original.append(self.y_test[i])
                    category_target_predicted_before.append(y_predicted_before[i])
                    category_target_predicted_after.append(y_predicted_after[i])
                    #print(y_predicted[i], len(self.y_test[i]), self.y_test[i], self.y_test)
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
                    if runkl:
                        a_before_sub.append(p_pred[i][0])
                        b_before_sub.append(p_pred[i][1])
                        #if q_pred != None:
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
            TPR_after = calculate_TPR(TP_after, FP_after, TN_after, FN_after)
            FPR_after = calculate_FPR(TP_after, FP_after, TN_after, FN_after)
            SP_after = calculate_SP(TP_after, FP_after, TN_after, FN_after)

            proportion = np.sum(
                category_target_predicted_before != category_target_predicted_after)  # calculate_proportion(TP, FP, TN, FN)
            kl_pos = None
            kl_neg = None
            if val[0] != val[1]:
                category = str(val[0]) + '-' + str(val[1])

                if runkl:
                    kl_pos, p, q, lin = run_dl_divergence(a_before_sub, a_after_sub, 'Pos_before', 'Pos_after',
                                                                   column_index=column_id, sub_category=self.colums_list[column_id]+' ({})_Pos'.format(category), path=self.new_path)
                    #print('kl test: ', len(b_before_sub), len(b_after_sub), b_before_sub, b_after_sub)
                    kl_neg, p2, q2, lin2 = run_dl_divergence(b_before_sub, b_after_sub, 'Neg_before', 'Neg_after',
                                                                      column_index=column_id, sub_category=self.colums_list[column_id]+' ({})_Neg'.format(category), path=self.new_path)
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
                data_category[category]['KL_Pos'] = kl_pos
                data_category[category]['KL_Neg'] = kl_neg
        return data_category

    def fit_nn(self):
        y_train = to_categorical(self.y_train)
        y_test = to_categorical(self.y_test)
        n_classes = y_test.shape[1]
        NNmodel = NNClassifier(self.x_train.shape[1], n_classes)
        NNmodel.fit(self.x_train, y_train)

        p_pred = NNmodel.predict(self.x_test)
        a_before, b_before = [], []
        list_before, list_after = [], []
        for pred_ in p_pred:
            #print('predicted: ', np.around(pred_))
            a_before.append(pred_[0])
            b_before.append(pred_[1])

            arg_max = pred_.argmax(axis=-1)
            list_before.append(arg_max)

        #for column_id in range(self.x_train.shape[1]):
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
            #if column_id != self.sensitive_index:
            #print('Column type name: ',self.data[0:, column_id].dtype.name)
            a_after, b_after = [], []
            if is_categorical(self.x_test,column_id):
                x_test_altered = alter_feature_values_categorical(self.x_test, self.get_categorical_features(column_id),
                                                                  column_id)
                #q_pred = NNmodel.predict(tf.convert_to_tensor(x_test_altered, dtype=tf.float32))
                q_pred = NNmodel.predict(x_test_altered)
                list_after = []
                for pred_ in q_pred:
                    a_after.append(pred_[0])
                    b_after.append(pred_[1])
                    arg_max = pred_.argmax(axis=-1)
                    list_after.append(arg_max)
                data_category_after = self.compute_metrics(list_before, list_after, column_id, p_pred, q_pred, runkl=True)

            else:
                x_test_altered = alter_feature_value_continous(self.x_test, self.determine_range(column_id),
                                                               column_id)
                # q_pred = NNmodel.predict(tf.convert_to_tensor(x_test_altered, dtype=tf.float32))
                q_pred = NNmodel.predict(x_test_altered)
                list_after = []
                for pred_ in q_pred:
                    a_after.append(pred_[0])
                    b_after.append(pred_[1])
                    arg_max = pred_.argmax(axis=-1)
                    list_after.append(arg_max)
                data_category_after = self.compute_metrics_continous(list_before, list_after, column_id, p_pred,q_pred, runkl=True)

            for key, val in data_category_after.items():
                self.data_writer2.writerow(
                    ['F_{}'.format(column_id), self.colums_list[column_id],key, data_category_after[key]['ACC'], data_category_after[key]['ACC_after'], data_category_after[key]['KL_Pos'], data_category_after[key]['KL_Neg'], data_category_after[key]['proportion'], data_category_after[key]['SP'], data_category_after[key]['SP_after'], data_category_after[key]['TPR'], data_category_after[key]['TPR_after'], data_category_after[key]['FPR'], data_category_after[key]['FPR_after']])

            kl_div_positive, p, q, lin = run_dl_divergence(a_before, a_after, 'Pos_before', 'Pos_after', column_index=column_id, show_figure=True, sub_category=self.colums_list[column_id]+'_Pos', path=self.new_path)
            kl_div_negative, p2, q2, lin2 = run_dl_divergence(b_before, b_after, 'Neg_before', 'Neg_after', column_index=column_id, show_figure=True, sub_category=self.colums_list[column_id]+'_Neg',path=self.new_path)

            #self.data_writer2.writerow(
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
                for i in range(len(self.x_test)):
                    # print(y_predicted[i], len(self.y_test[i]), self.y_test[i], self.y_test)
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
                #category_target_original = np.array(category_target_original)
                list_before = np.array(list_before)
                list_after = np.array(list_after)
                acc_category = round((np.sum(self.y_test == list_before)*100/len(self.y_test)),3)
                TPR = calculate_TPR(TP, FP, TN, FN)
                FPR = calculate_FPR(TP, FP, TN, FN)
                SP = calculate_SP(TP, FP, TN, FN)

                casuality = round((np.sum(list_before != list_after)*100/len(self.y_test)),3)

                acc_category_after = round((np.sum(self.y_test == list_after)*100/len(self.y_test)),3)
                TPR_after = calculate_TPR(TP_after, FP_after, TN_after, FN_after)
                FPR_after = calculate_FPR(TP_after, FP_after, TN_after, FN_after)
                SP_after = calculate_SP(TP_after, FP_after, TN_after, FN_after)

                self.data_writer.writerow(
                    ['F_{}'.format(column_id), self.colums_list[column_id], acc_category, acc_category_after, kl_div_positive, kl_div_negative, casuality, SP, SP_after, TPR, TPR_after, FPR,
                     FPR_after])

                #self.data_writer2.writerow(['F_{}'.format(column_id), self.colums_list[column_id], 'NEG', kl_div_negative, total_equal_neg/len(q_pred), total_equal_all])

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
    #target_column = 'Probability'
    correlation_threshold = 0.45 #0.35
    loadData = LoadData(path,threshold=correlation_threshold) #,threshold=correlation_threshold
    data_name = 'adult-45' #_35_threshold

    #df_adult = loadData.load_adult_data('adult.data.csv')
    df_adult = loadData.load_credit_defaulter('default_of_credit_card_clients.csv')
    #df_adult = loadData.load_student_data('Student.csv')
    #df_adult = loadData.load_student_data('Student.csv')
    sensitive_list = loadData.sensitive_list
    sensitive_indices = loadData.sensitive_indices
    colums_list = df_adult.columns.tolist()


    #corr_columns = get_correlation(df_adult, correlation_threshold)

    #corrMatrix = df_adult.corr()
    #sn.heatmap(corrMatrix, annot=True)
    #corr = df_adult.corr()
    #fig, ax = plt.subplots(figsize=(10, 10))
    #ax.matshow(corr)
    #plt.xticks(range(len(corr.columns)), corr.columns)
    #plt.yticks(range(len(corr.columns)), corr.columns)


    #plt.savefig(path + "/png/{}.png".format(data_name))
    #plt.show()
    target_name = loadData.target_name
    target_index = loadData.target_index
    for colum in colums_list:
        print(colum, df_adult[target_name].corr(df_adult[colum]))
    #corr = df_adult.corr()
    #corr.style.background_gradient(cmap='coolwarm')

    df_adult.to_csv(path+'{}-transformed.csv'.format(data_name), index=False)

    #print(df_adult)
    #print(target_name)
    #print(target_index)
    sensitivityAnalysis = SensitivityAnalysis(df_adult.to_numpy(), target_index=target_index, sensitive_name=sensitive_list[0], sensitive_index=sensitive_indices[0], data_name=data_name, colums_list=colums_list, threshold=correlation_threshold)
    sensitivityAnalysis.fit_nn()


