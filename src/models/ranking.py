import numpy as np
import pandas as pd
import operator

from tensorflow.python.keras.utils.np_utils import to_categorical
import xgboost
import shap
import sklearn
from src.models.v2.utils import data_split
from src.models.v3.NNClassifier import NNClassifier
from src.models.v3.load_data import LoadData
from src.models.v3.utility_functions import *
from shapley import PermutationSampler
import sage
class Ranking:
    def __init__(self):
        pass
    def sort_dict(self, dict_, reverse=True):
        return dict(sorted(dict_.items(),key=operator.itemgetter(1),reverse=reverse))
    def check_similarity(self, dict_):
        val_dict = {}
        val_dict2 = {}
        for key, val in dict_.items():
            if isinstance(val,list) or isinstance(val,tuple):
                if val[0] in val_dict.keys():
                    val_dict[val[0]].append(key)
                else:
                    val_dict[val[0]] = [key]
                if val[1] in val_dict2.keys():
                    val_dict2[val[1]].append(key)
                else:
                    val_dict2[val[1]] = [key]
            else:
                if val in val_dict.keys():
                    val_dict[val].append(key)
                else:
                    val_dict[val] = [key]
        return val_dict, val_dict2
    def rank_(self, val_dict, feature_ranking_data, features, type='mean'):
        dict_match = self.check_similarity(val_dict)
        sorted_dict_1 = self.sort_dict(val_dict, reverse=True)

        if type == 'mean':
            dict_match = dict_match[0]
        else:
            dict_match = dict_match[1]

            print('type: ', type)
            #if len(dict_match) == 0:
            #    dict_match = dict_match[0]
        # print(sorted_dict_1)
        new_dict = {}
        list_checked = []
        for key2, val2 in sorted_dict_1.items():
            list_features = dict_match.get(val2)
            #print('dict_match: ', val2, dict_match)
            if list_features != None:
                if len(list_features) > 1 and not key2 in list_checked:
                    list_checked.extend(list_features)
                    new_dict['/'.join(list_features)] = val2
        for key_pop in list_checked:
            val_dict.pop(key_pop)
        for key2, val2 in new_dict.items():
            if len(key2.split('/')) != len(set(features)):
                val_dict[key2] = val2
        sorted_dict = self.sort_dict(val_dict, reverse=True)
        rank = 1
        for key, val in sorted_dict.items():
            if not '/' in key:
                if key in feature_ranking_data.items():
                    feature_ranking_data[key].append(rank)
                else:
                    feature_ranking_data[key] = [rank]
            else:
                for feat in key.split('/'):
                    if feat in feature_ranking_data.items():
                        feature_ranking_data[feat].append(rank)
                    else:
                        feature_ranking_data[feat] = [rank]
            rank += 1

        #print(key, sorted_dict_1, sorted_dict)

    def ranks(self, psa_data, features, rank_type=None):
        feature_ranking_data = {}
        feature_ranking_data2 = {}
        ranks_metrics = 'Both'
        for key, val in psa_data.items():
            val_dict = {}
            val_dict2 = {}
            if 'mean' == rank_type:
                for k, v in val.items():
                    val_dict[k] = v[0]
                self.rank_(val_dict, feature_ranking_data, features, rank_type)
                ranks_metrics = 'Mean'
            elif 'median' == rank_type:
                for k, v in val.items():
                    val_dict[k] = v[1]
                self.rank_(val_dict, feature_ranking_data, features, rank_type)
                ranks_metrics = 'Median'
            else:
                for k, v in val.items():
                    val_dict[k] = v[0]
                    val_dict2[k] = v[1]
                #print('val_dict: ', val_dict)
                self.rank_(val_dict, feature_ranking_data, features, 'mean')
                self.rank_(val_dict2, feature_ranking_data2, features, 'median')
                ranks_metrics = 'Mean/Median'


            ##TODO: call the ranker here
        print("**** Ranking features ******")
        for key, val in feature_ranking_data.items():
            feature_ranking_data[key] = np.mean(val)
            # print(key, np.mean(val))
        for key, val in feature_ranking_data2.items():
            feature_ranking_data2[key] = np.mean(val)
        return feature_ranking_data, feature_ranking_data2 #, ranks_metrics
    def rank_average(self, df, feature_name, sub_category=None, PSA=[], rank_type='mean/median'):
        if sub_category != None:
            features = []
            for i in range(len(df)):
                features.append(str(df[feature_name].values[i])+'|'+str(df[sub_category].values[i]))
        else:
            features = df[feature_name].values.tolist()
        psa_data = {}
        for psa in PSA:
            for index in range(len(df)):
                if psa in psa_data.keys():
                    if features[index] in psa_data[psa].keys():
                        psa_data[psa][features[index]].append(df[psa][index])
                    else:
                        psa_data[psa][features[index]] = [df[psa][index]]
                else:
                    psa_data[psa] = {}
                    psa_data[psa][features[index]] = [df[psa][index]]

        for key, val in psa_data.items():

            for key2, val2 in val.items():
                psa_data[key][key2] = [np.mean(val2), np.median(val2)]

        #features_set = set(features)
        feature_ranking_data, feature_ranking_data2 = self.ranks(psa_data, features)
        return psa_data, feature_ranking_data, feature_ranking_data2
    def rank_multiple(self, ranking_data, rank_type='mean/median'):
        val_dict = {}
        val_dict2 = {}
        if 'mean' == rank_type:
            for k, v in ranking_data.items():
                val_dict[k] = v[0]
            ranks = self.sort_dict(val_dict, reverse=True) # self.rank_(val_dict, feature_ranking_data, features)
            ranks_metrics = 'Mean'
        elif 'median' == rank_type:
            for k, v in ranking_data.items():
                val_dict[k] = v[1]
            ranks = self.sort_dict(val_dict, reverse=True)
            #self.rank_(val_dict, feature_ranking_data, features)
            ranks_metrics = 'Median'
        else:
            for k, v in ranking_data.items():
                val_dict[k] = v[0]
                val_dict2[k] = v[1]
            #self.rank_(val_dict, feature_ranking_data, features)
            #self.rank_(val_dict2, feature_ranking_data2, features)
            ranks_metrics = 'Mean/Median'
        ranks1 = self.sort_dict(val_dict, reverse=True)
        ranks2 = self.sort_dict(val_dict2, reverse=True)
        rank_id = 1
        for key, val in ranks1.items():
            ranks1[key] = rank_id
            rank_id += 1
        rank_id2 = 1
        for key, val in ranks2.items():
            ranks1[key] = rank_id2
            rank_id2 += 1
        return ranking_data, ranks1, ranks2

    def rank_multiplicative_score_function(self, df, feature_name, sub_category=None, PSA=[], rank_type='mean/median'):
        #todo: https://pubsonline.informs.org/doi/pdf/10.1287/ited.2013.0124
        if sub_category != None:
            features = []
            for i in range(len(df)):
                features.append(str(df[feature_name].values[i])+'|'+str(df[sub_category].values[i]))
        else:
            features = df[feature_name].values.tolist()
        psa_data = {}
        # Multiplicative score function
        for index in range(len(df)):
            multi_val = 1
            for psa in PSA:
                multi_val *= df[psa][index]
            if features[index] in psa_data.keys():
                psa_data[features[index]].append(multi_val)
            else:
                psa_data[features[index]] = [multi_val]
        for key, val in psa_data.items():
            psa_data[key] = [np.mean(val), np.median(val)]
        return self.rank_multiple(psa_data,rank_type)


class Shapley:
    def __init__(self, data, target_index, column_list):
        self.data = data
        self.target_index = target_index
        self.data_init(data)
        self.column_list = column_list

    def data_init(self, data):
        self.train, self.test = data_split(data=data, sample_size=0.25)
        self.x_train, self.y_train = split_features_target(self.train, index=self.target_index)
        self.x_test, self.y_test = split_features_target(self.test, index=self.target_index)

        #print('self.y_test: ', self.y_test)
    def fit_model(self, feature_names):
        y_train = to_categorical(self.y_train)
        y_test = to_categorical(self.y_test)
        n_classes = y_test.shape[1]
        NNmodel = NNClassifier(self.x_train.shape[1], n_classes)

        NNmodel.fit(self.x_train, y_train)

        # Set up an imputer to handle missing features
        #imputer = sage.MarginalImputer(NNmodel, self.x_test)

        # Set up an estimator
        #estimator = sage.PermutationEstimator(imputer, 'mse')

        # Calculate SAGE values
        #sage_values = estimator(self.x_test, self.y_test)
        #sage_values.plot(feature_names)

        # explain the model's predictions using SHAP
        # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
        model = xgboost.XGBRegressor().fit(self.x_train, self.y_train)
        pred = model.predict(self.x_test)

        #print('xgboost regressor: ', pred)

        knn_norm = sklearn.neighbors.KNeighborsRegressor()
        knn_norm.fit(self.x_train, self.y_train)
        pred = knn_norm.predict(self.x_test)
        #print('xgboost regressor: ', pred)
        f = lambda x: knn_norm.predict(x)#[:, 1]
        #med = self.x_train.reshape((1, self.x_train.shape[1]))
        explainer = shap.KernelExplainer(f, self.x_train[:10,:])
        #shap_values_single = explainer.shap_values(self.x_test.iloc[0, :], nsamples=1000)
        shap_values_single = explainer.shap_values(self.x_test[:2,:])


        print('shap_values_single: ', shap_values_single)
        for i in range(len(shap_values_single)):
            print('\nFirst row: ')
            shap_dict = {}
            for row_id in range(len(shap_values_single[i])):
                #print(row_id, self.column_list[row_id], shap_values_single[i][row_id])
                shap_dict[self.column_list[row_id]] = abs(shap_values_single[i][row_id])
            print(sort_dict(shap_dict, reverse=True))
        #explainer = shap.Explainer(model)
        #shap_values = explainer(self.x_test[0])

        # visualize the first prediction's explanation
        #shap.plots.waterfall(shap_values[0])
        #shap.initjs()
        f = lambda x: model.predict(x)  # [:, 1]
        # we use the first 100 training examples as our background dataset to integrate over
        explainer = shap.PermutationExplainer(model.predict, self.x_test)

        # explain the first 10 predictions
        # explaining each prediction requires 2 * background dataset size runs
        #shap_values = explainer.shap_values(self.x_test)
        shap_values = explainer(self.x_test, max_evals=10 * self.x_test.shape[1], main_effects=False)
        print('Permutation values: ', shap_values)
        '''for i in range(len(shap_values)):
            print('\nFirst row: ')
            shap_dict = {}
            for row_id in range(len(shap_values[i])):
                # print(row_id, self.column_list[row_id], shap_values_single[i][row_id])
                shap_dict[self.column_list[row_id]] = abs(shap_values[i][row_id])
            print(sort_dict(shap_dict, reverse=True))'''

        # we use the first 100 training examples as our background dataset to integrate over
        explainer = shap.PermutationExplainer(NNmodel.predict, self.x_test)

        # explain the first 10 predictions
        # explaining each prediction requires 2 * background dataset size runs
        # shap_values = explainer.shap_values(self.x_test)
        shap_values = explainer(self.x_test, max_evals=10 * self.x_test.shape[1], main_effects=False)
        print('Neural Network and Permutation values: ', shap_values)
        '''for i in range(len(shap_values)):
            print('\nFirst row: ')
            shap_dict = {}
            for row_id in range(len(shap_values[i])):
                # print(row_id, self.column_list[row_id], shap_values_single[i][row_id])
                shap_dict[self.column_list[row_id]] = abs(shap_values[i][row_id])
            print(sort_dict(shap_dict, reverse=True))'''
        #shap.plots.bar(shap_values)
        shap.summary_plot(shap_values, self.x_test, plot_type="layered_violin", color='#cccccc')





path = '../dataset/'
if __name__ == '__main__':
    #df = pd.read_csv(path+'logging/log_KL_Divergence_overral_adult-45.csv')
    df_global = pd.read_csv(path + 'logging2/log_KL_Divergence_overral_adult-45_2_50.csv')
    ranking_attributes = ['JS_Divergence', 'Casuality',  'Importance'] # , 'SP', 'Casuality',
    #ranking_attributes = ['JS_Divergence', 'Casuality']  # , 'SP', 'Casuality',
    PSA = 'Feature'
    sub_category = 'Category'
    ranking = Ranking()
    rank_global = ranking.rank_average(df_global,feature_name=PSA, PSA=ranking_attributes)
    rank_global = ranking.sort_dict(rank_global, reverse=False)
    print('Global ranking: ', rank_global)

    #df = pd.read_csv(path + 'logging/log_kl_divergence_adult-45.csv')
    df_local = pd.read_csv(path + 'logging2/log_kl_divergence_adult-45_2_50.csv')
    rank_local = ranking.rank_average(df_local, PSA, sub_category=sub_category, PSA=ranking_attributes)
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


    ## Rank by multiplicative score function
    print('\n***** Ranking by Multiplicative Score Function ********\n')
    rank_global = ranking.rank_multiplicative_score_function(df_global, feature_name=PSA, PSA=ranking_attributes)
    rank_global = ranking.sort_dict(rank_global, reverse=False)
    print('Global ranking MSF: ', rank_global)

    rank_local = ranking.rank_multiplicative_score_function(df_local, PSA, sub_category=sub_category, PSA=ranking_attributes)
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
    print('Local ranking MSF: ', ranking.sort_dict(data_ranks, reverse=False))


    #todo: test shaply and sage method
    '''correlation_threshold = 0.45  # 0.35
    loadData = LoadData(path, threshold=correlation_threshold)  # ,threshold=correlation_threshold
    data_name = 'student-45'  # _35_threshold

    #df_adult = loadData.load_adult_data('adult.data.csv')
    # df_adult = loadData.load_credit_defaulter('default_of_credit_card_clients.csv')
    df_adult = loadData.load_student_data('Student.csv')
    # df_adult = loadData.load_student_data('Student.csv')
    sensitive_list = loadData.sensitive_list
    sensitive_indices = loadData.sensitive_indices
    colums_list = df_adult.columns.tolist()
    target_index = loadData.target_index

    #shapley = Shapley(df_adult.to_numpy(), target_index, colums_list)
    #shapley.shapley()
    #shapley.fit_model(df_adult.columns.tolist())'''





