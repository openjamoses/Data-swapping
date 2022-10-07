import csv
import os

import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn import model_selection

from src.models.data.models import Models2
from src.models.v3.load_data import LoadData
from src.models.v3.utility_functions import *
import shap
def global_shap_importance(model, X):
    """ Return a dataframe containing the features sorted by Shap importance
    Parameters
    ----------
    model : The tree-based model
    X : pd.Dataframe
         training set/test set/the whole dataset ... (without the label)
    Returns
    -------
    pd.Dataframe
        A dataframe containing the features sorted by Shap importance
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    cohorts = {"": shap_values}
    cohort_labels = list(cohorts.keys())
    cohort_exps = list(cohorts.values())
    for i in range(len(cohort_exps)):
        if len(cohort_exps[i].shape) == 2:
            cohort_exps[i] = cohort_exps[i].abs.mean(0)
    features = cohort_exps[0].data
    feature_names = cohort_exps[0].feature_names
    values = np.array([cohort_exps[i].values for i in range(len(cohort_exps))])
    feature_importance = pd.DataFrame(
        list(zip(feature_names, sum(values))), columns=['features', 'importance'])
    feature_importance.sort_values(
        by=['importance'], ascending=False, inplace=True)
    return feature_importance
class SharpAnalysis:
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

    def data_init(self, data):
        self.train, self.test = data_split(data=data, sample_size=0.25)
        self.x_train, self.y_train = split_features_target(self.train, index=self.target_index)
        self.x_test, self.y_test = split_features_target(self.test, index=self.target_index)

        print('self.y_test: ', self.y_test)
    def fit_shap(self):
        X, Y = split_features_target(self.data, index=self.target_index)
        # kf = model_selection.KFold(n_splits=10)
        kf = model_selection.StratifiedKFold(n_splits=10)
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



            #explainer = shap.PermutationExplainer(NNmodel.predict, self.x_train[:10, :])

            # explain the first 10 predictions
            # explaining each prediction requires 2 * background dataset size runs
            # shap_values = explainer.shap_values(self.x_test)
            #shap_values = explainer(self.x_test[:5, :], max_evals=10 * self.x_test.shape[1], main_effects=False)

            #print('shap_values.data:', shap_values.data)
            #print('shap_values.values: ', shap_values.values)
            #print('shap_values.base_values: ', shap_values.base_values)
            #print('shap_values.feature_names: ', shap_values.feature_names)
            explainer = shap.TreeExplainer(NNmodel)
            shap_values = explainer.shap_values(self.x_test)
            print(len(shap_values))

            #for shap_v in shap_values:
            #    #print(colums_list[shap_v)
            #    print(np.mean(abs(shap_v[0:, 0])), np.mean(shap_v[0:, 1]), np.mean(shap_v[0:, 2]), np.mean(shap_v[0:, 3]), np.mean(shap_v[0:, 4]))
            #print(shap_values)
            #cohorts = shap_values.cohorts
            features_ = [colums_list[x] for x in range(len(colums_list)-1)]
            vals = np.abs(shap_values).mean(0)
            for f_name, val, fImport in zip(features_, sum(vals), NNmodel.feature_importances_):
                print(f_name, val, fImport)
            #for i, v in zip(features_, NNmodel.feature_importances_):
            #    print('Feature: ',i, 'Score: %.5f' % (v))
            print('NNmodel.feature_importances_: ', NNmodel.feature_importances_)
            feature_importance = pd.DataFrame(list(zip(features_, sum(vals))),
                                              columns=['col_name', 'feature_importance_vals'])
            feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
            #feature_importance.head()

            #print(global_shap_importance(NNmodel, self.x_test))


            #cohorts = {"": shap_values}
            # unpack our list of Explanation objects we need to plot
            #cohort_labels = list(cohorts.keys())
            #cohort_exps = list(cohorts.values())
            #print(explainer.feature_names)
            #shap.summary_plot(shap_values, self.x_test)

            break
if __name__ == '__main__':
    path = '../dataset/'
    ### Adult dataset
    # target_column = 'Probability'
    fair_error = 0.01
    alpha = 0.3
    correlation_threshold = 0.45  # 0.35
    loadData = LoadData(path, threshold=correlation_threshold)  # ,threshold=correlation_threshold
    data_name = 'compas-shap-{}_'.format(alpha)  # _35_threshold

    #df_adult = loadData.load_adult_data('adult.data.csv')
    # df_adult = loadData.load_credit_defaulter('default_of_credit_card_clients.csv')
    #df_adult = loadData.load_clevelan_heart_data('processed.cleveland.data.csv')
    #df_adult = loadData.load_student_data('Student.csv')
    #df_adult = loadData.load_german_data2('german_credit_data.csv')
    #df_adult = loadData.load_german_data('GermanData.csv')
    df_adult = loadData.load_compas_data('compas-scores-two-years.csv')
    sensitive_list = loadData.sensitive_list
    sensitive_indices = loadData.sensitive_indices
    colums_list = df_adult.columns.tolist()

    print('Features: ', colums_list[:-1])
    target_name = loadData.target_name
    target_index = loadData.target_index
    shapAnalysis = SharpAnalysis(df_adult.to_numpy(), df_adult, target_index=target_index,
                                              sensitive_name=sensitive_list[0], sensitive_index=sensitive_indices[0],
                                              data_name=data_name, colums_list=colums_list,
                                              threshold=correlation_threshold)
    shapAnalysis.fit_shap()
