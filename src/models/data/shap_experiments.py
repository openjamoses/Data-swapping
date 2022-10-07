import csv
import numpy as np
import shap
import os
import time
from src.models.data.data_range import get_categorical_features, determine_range, check_member_continous
from src.models.data.load_train_test import LoadTrainTest
from src.models.data.models import Models
from src.models.v3.sensitivity_utils import is_categorical
from src.models.v3.utility_functions import split_features_target
from pathlib import Path

class ShapExperiment:
    def __init__(self, train, test, target_index, column_list, data_path, data_name, normalize):
        self.target_index = target_index
        self.column_list = column_list
        self.x_train, self.y_train = split_features_target(train, index=self.target_index)
        self.x_test, self.y_test = split_features_target(test, index=self.target_index)
        self.model_class = Models(train,test, self.target_index, normalize=normalize)
        self.root_path = data_path+data_name+'/'
        self.data_name = data_name
        self.normalize = normalize
        if normalize:
            self.normalise_data()
    def normalise_data(self):
        # normalize data
        self.x_train_norm = (self.x_train - self.x_train.mean(axis=0)) / self.x_train.std(axis=0)
        self.x_train_norm.mean(axis=0)

        self.x_test_norm = (self.x_test - self.x_test.mean(axis=0)) / self.x_test.std(axis=0)
        self.x_test_norm.mean(axis=0)
    def add_dict(self, key_, shap_index, j, data_shap_values, shap_values):
        if self.column_list[shap_index] in data_shap_values.keys():
            if key_ in data_shap_values[self.column_list[shap_index]].keys():
                data_shap_values[self.column_list[shap_index]][key_].append(
                    shap_values[j][shap_index])
            else:
                data_shap_values[self.column_list[shap_index]][key_] = [
                    shap_values[j][shap_index]]
        else:
            data_shap_values[self.column_list[shap_index]] = {}
            data_shap_values[self.column_list[shap_index]][key_] = [
                shap_values[j][shap_index]]

    def write_csv(self, model_name, shap_values):
        if not os.path.exists(self.root_path + 'shap_values/global'):
            Path(self.root_path + 'shap_values/global/').mkdir(parents=True)
        if not os.path.exists(self.root_path + 'shap_values/local'):
            Path(self.root_path + 'shap_values/local/').mkdir(parents=True)
        if not os.path.exists(self.root_path + 'shap_values/log'):
            Path(self.root_path + 'shap_values/log/').mkdir(parents=True)
        data_file = open(self.root_path + 'shap_values/log/log_shap_values_{}_{}.csv'.format(self.data_name, model_name), mode='w', newline='',
                               encoding='utf-8')
        data_writer = csv.writer(data_file)
        data_writer.writerow(self.column_list[:-1])
        for sha_row in shap_values:
            data_writer.writerow(list(sha_row))
        data_file_global = open(self.root_path + 'shap_values/global/global_shap_values_{}_{}.csv'.format(self.data_name, model_name),
                         mode='w', newline='',
                         encoding='utf-8')
        data_writer_global = csv.writer(data_file_global)
        data_writer_global.writerow(['Feature', 'Shap_value_mean', 'Shap_value_median'])

        data_file_local = open(
            self.root_path + 'shap_values/local/local_shap_values_{}_{}.csv'.format(self.data_name, model_name),
            mode='w', newline='',
            encoding='utf-8')
        data_writer_local = csv.writer(data_file_local)
        data_writer_local.writerow(['Feature', 'Category', 'Shap_value_mean', 'Shap_value_median'])

        data_shap_values = {}
        for shap_index in range(len(self.column_list)-1):
            if is_categorical(self.x_test, shap_index):
                for j in range(len(shap_values)):
                    self.add_dict(self.x_test[j][shap_index], shap_index,j,data_shap_values, shap_values)
            else:
                data_range_folder = determine_range(self.x_train, shap_index)
                for j in range(len(shap_values)):
                    key_returned = check_member_continous(data_range_folder, self.x_test[j][shap_index])
                    self.add_dict(key_returned, shap_index, j, data_shap_values, shap_values)
            print(shap_values[:,shap_index])
            data_writer_global.writerow([self.column_list[shap_index], np.mean(shap_values[:,shap_index]), np.median(shap_values[:,shap_index])])
        for key, val in data_shap_values.items():
            for key2, val2 in val.items():
                data_writer_local.writerow([key, key2, np.mean(val2), np.median(val2)])
        data_file.close()
        data_file_global.close()
        data_file_local.close()

    def permuation(self):
        # todo: gaussianNB
        since = time.time()
        model_name = 'gaussianNB'
        model = self.model_class.gaussianNB()
        # pred = self.model_class.model_predict()
        # self.model_class.accuracy(self.y_test, pred)
        explainer = shap.PermutationExplainer(model.predict_proba, self.x_train)
        shap_values = explainer(self.x_test[:10], max_evals=10 * self.x_test.shape[1], main_effects=False)
        self.write_csv(model_name, np.absolute(shap_values.values))
        time_elapsed_1 = time.time() - since
        print(model_name, '{:.0f}m {:.0f}s'.format(time_elapsed_1 // 60, time_elapsed_1 % 60))

        #todo: xgboost regression
        '''since = time.time()
        model_name = 'xgboost_regression'
        model = self.model_class.xgboost_regressor()
        pred = self.model_class.model_predict()
        self.model_class.accuracy(self.y_test, pred)

        explainer = shap.PermutationExplainer(model.predict_proba, self.x_train)
        #shap_values = explainer(self.x_test)
        shap_values = explainer(self.x_test, max_evals=10 * self.x_test.shape[1], main_effects=False)
        shap_valuesL = []

        #print(shap_values.values[:, :, 1])
        shap_val = shap_values.values[:, :, 1]
        for shap_id in range(len(shap_val)):
            print(shap_id, shap_val[shap_id])
        self.write_csv(model_name, np.absolute(shap_values.values))
        time_elapsed_1 = time.time() - since
        print(model_name, '{:.0f}m {:.0f}s'.format(time_elapsed_1 // 60, time_elapsed_1 % 60))'''


        # todo: knn_regressor
        since = time.time()
        model_name = 'knn_regressor'
        model = self.model_class.knn_regressor()
        explainer = shap.PermutationExplainer(model.predict, self.x_train)
        shap_values = explainer(self.x_test[:10], max_evals=10 * self.x_test.shape[1], main_effects=False)
        self.write_csv(model_name, shap_values.values)
        time_elapsed_1 = time.time() - since
        print(model_name, '{:.0f}m {:.0f}s'.format(time_elapsed_1 // 60, time_elapsed_1 % 60))

        # todo: decision_tree_regressor
        since = time.time()
        model_name = 'decision_tree_regressor'
        model = self.model_class.decision_tree_regressor()
        #pred = self.model_class.model_predict()
        #self.model_class.accuracy(self.y_test, pred)

        explainer = shap.PermutationExplainer(model.predict_proba, self.x_train)
        #shap_values = explainer(self.x_test)

        shap_values = explainer(self.x_test[:10], max_evals=10 * self.x_test.shape[1], main_effects=False)
        self.write_csv(model_name, np.absolute(shap_values.values))
        time_elapsed_1 = time.time() - since
        print(model_name, '{:.0f}m {:.0f}s'.format(time_elapsed_1 // 60, time_elapsed_1 % 60))

        # todo: logistic_regression
        since = time.time()
        model_name = 'logistic_regression'
        model = self.model_class.logistic_regression()
        #pred = self.model_class.model_predict()
        #self.model_class.accuracy(self.y_test, pred)
        explainer = shap.PermutationExplainer(model.predict_proba, self.x_train)
        #shap_values = explainer(self.x_test)
        shap_values = explainer(self.x_test[:10], max_evals=10 * self.x_test.shape[1], main_effects=False)
        self.write_csv(model_name, np.absolute(shap_values.values))
        time_elapsed_1 = time.time() - since
        print(model_name, '{:.0f}m {:.0f}s'.format(time_elapsed_1 // 60, time_elapsed_1 % 60))
        #shap_val = shap_values.values[:, :, 1]
        #print(shap_val)
        #for shap_id in range(len(shap_val)):
        #    print(shap_id, shap_val[shap_id])

        # todo: random_forest
        '''since = time.time()
        model_name = 'random_forest'
        model = self.model_class.random_forest()
        explainer = shap.PermutationExplainer(model.predict, self.x_train)
        shap_values = explainer(self.x_test)
        self.write_csv(model_name, np.absolute(shap_values.values))
        time_elapsed_1 = time.time() - since
        print(model_name, '{:.0f}m {:.0f}s'.format(time_elapsed_1 // 60, time_elapsed_1 % 60))'''

        # todo: nn_classifier
        since = time.time()
        model_name = 'nn_classifier'
        #self.model_class = Models(train, test, self.target_index, normalize=self.normalize)
        model = self.model_class.nn_classifier()
        #pred = self.model_class.model_predict()
        #self.model_class.accuracy(self.y_test, pred)
        explainer = shap.PermutationExplainer(model.predict, self.x_train)
        shap_values = explainer(self.x_test[:10], max_evals=10 * self.x_test.shape[1], main_effects=False)
        self.write_csv(model_name, np.absolute(shap_values.values[:, :, 1])) #
        time_elapsed_1 = time.time() - since
        print(model_name, '{:.0f}m {:.0f}s'.format(time_elapsed_1 // 60, time_elapsed_1 % 60))
        #shap_val = shap_values.values #[:, :, 1]
        #print(shap_val)
        #print('base_values: ', shap_values.base_values)
        #print('abs: ', shap_values.abs)




        # explain the first 10 predictions
        # explaining each prediction requires 2 * background dataset size runs
        #shap_values = explainer.shap_values(self.x_test,main_effects=False)
        #shap_values = explainer(self.x_test)
        #print('shap_values.shape: ', shap_values.shape)
        #print('shap_values results: ', shap_values.values[:, :, 1])'''
        '''print('shap: ', shap_values)
        print('shap_values.feature_names: ', shap_values.feature_names)
        print('shap_values.data: ', shap_values.data)
        print('shap_values.values: ', shap_values.values)
        print('shap_values.base_values: ', shap_values.base_values)
        shap.summary_plot(shap_values, self.x_test)'''

        #for coumn_index in range(len(self.column_list)-1):


        # explain all the predictions in the test set
        #explainer = shap.PermutationExplainer(model.predict, self.x_train)
        #shap_values = explainer.shap_values(self.x_test[:100])
        #shap.summary_plot(shap_values, self.x_test)
        #f = lambda x: model.predict(x)
        #print(shap_values)

        #explainer = shap.Explainer(f, self.x_train_norm)

        #shap_values_norm = explainer(self.x_test_norm)
        #print('shap_values_norm: ', shap_values_norm)
if __name__ == '__main__':
    data_name = 'adult'
    path = '../dataset/experiments/data/'
    load_data = LoadTrainTest(path)
    train, test, target_index = load_data.load_adult(data_name)

    shapExperiment = ShapExperiment(train.to_numpy(),test.to_numpy(),target_index, train.columns.tolist(), data_path=path, data_name=data_name, normalize=True)
    shapExperiment.permuation()
