from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler

from imblearn.combine import SMOTEENN

import sys
from src.models.v2.utils import *
from src.models.v3.load_data import LoadData
from src.models.v3.utility_functions import *
sys.path.insert(0, '../../examples/fair_classification/') # the code for fair classification is in this directory
import utils as ut
import numpy as np
import loss_funcs as lf # loss funcs that can be optimized subject to various constraints

class ModelClass:
    def __init__(self, x_train, y_train, x_test, y_test, sensitive_index, protected=0, unprotected=1):
        self.x_train, self.y_train, self.x_test, self.y_test = x_train, y_train, x_test, y_test
        self.sensitive_index = sensitive_index
        self.protected = protected
        self.unprotected = unprotected
    def apply_casuality_train(self, model,system_type='All'):
        x_train2 = []
        y_train2 = []
        for i in range(len(self.x_train)):
            row = []
            for j in range(len(self.x_train[i])):
                temp_val = self.x_train[i][j]
                if self.x_train[i][j] == self.protected and j == self.sensitive_index:
                    temp_val = self.unprotected
                elif self.x_train[i][j] == self.unprotected and j == self.sensitive_index:
                    temp_val = self.protected
                row.append(temp_val)
            #row2 = [x for x in row[:-1]]
            if system_type == 'A':
                pred_original = np.sign(np.dot(model, self.x_train[i]))  # model.predict()
                pred_after = np.sign(np.dot(model, row))
            else:
                pred_original = model.predict(np.reshape(np.array(self.x_train[i]), [-1, self.x_train.shape[1]]))
                pred_after = model.predict(np.reshape(np.array(row), [-1, self.x_train.shape[1]]))
            if pred_original == pred_after:
                x_train2.append(self.x_train[i])
                y_train2.append(self.y_train[i])
        return x_train2, y_train2
    def apply_casuality_test(self, model,system_type='All'):
        x_test2 = []
        y_test2 = []
        for i in range(len(self.x_test)):
            row = []
            for j in range(len(self.x_test[i])):
                temp_val = self.x_test[i][j]
                if self.x_test[i][j] == self.protected and j == self.sensitive_index:
                    temp_val = self.unprotected
                elif self.x_test[i][j] == self.unprotected and j == self.sensitive_index:
                    temp_val = self.protected
                row.append(temp_val)
            #row2 = [x for x in row[:-1]]
            if system_type == 'A':
                pred_original = np.sign(np.dot(model, self.x_test[i]))  # model.predict()
                pred_after = np.sign(np.dot(model, row))
            else:
                pred_original = model.predict(np.reshape(np.array(self.x_test[i]), [-1, self.x_test.shape[1]]))
                pred_after = model.predict(np.reshape(np.array(row), [-1, self.x_test.shape[1]]))
            if pred_original == pred_after:
                x_test2.append(self.x_test[i])
                y_test2.append(self.y_test[i])
        return x_test2, y_test2





    def fit_GaussianNB(self):
        print('******** GaussianNB ****** ')
        clf = GaussianNB()
        clf.fit(self.x_train, self.y_train)
        y_pred = clf.predict(self.x_test)
        scores_metrics(y_pred, self.y_test)
        fairness_metrics(self.x_test, self.y_test, y_pred, self.sensitive_index, protected=self.protected,
                         unprotected=self.unprotected)
        return clf
    def fit_DecisionTree(self):
        print('******** DecisionTree Classifier ****** ')
        clf = DecisionTreeClassifier()
        clf.fit(self.x_train, self.y_train)
        y_pred = clf.predict(self.x_test)
        scores_metrics(y_pred, self.y_test)
        fairness_metrics(self.x_test, self.y_test, y_pred,self.sensitive_index, protected=self.protected, unprotected=self.unprotected)
        return clf
    def fit_RandomForest(self):
        print('******** RandomForest Classifier ****** ')
        clf = RandomForestClassifier()
        clf.fit(self.x_train, self.y_train)
        y_pred = clf.predict(self.x_test)
        scores_metrics(y_pred, self.y_test)
        fairness_metrics(self.x_test, self.y_test, y_pred, self.sensitive_index, protected=self.protected,
                         unprotected=self.unprotected)

        return clf

    def fit_LogisticRegression(self):
        print('******** Logistic Regression ****** ')
        clf = LogisticRegression(random_state=42, penalty='l2')
        clf.fit(self.x_train, self.y_train)
        y_pred = clf.predict(self.x_test)
        scores_metrics(y_pred, self.y_test)
        fairness_metrics(self.x_test, self.y_test, y_pred, self.sensitive_index, protected=self.protected,
                         unprotected=self.unprotected)
        return clf

    def fit_FairClassifier(self, sensitive_name, sensitive_index):
        print('******** Fair Classifier ****** ')
        if (sensitive_name == 'sex'):
            cov = 0
        else:
            cov = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

        sen = []
        for i in range(len(self.x_train)):
            sen.append(self.x_train[i][sensitive_index])
        sensitive = {}
        sens = []
        sensitive[sensitive_name] = np.array(sens, dtype=float)
        loss_function = lf._logistic_loss
        sep_constraint = 0
        sensitive_attrs = [sensitive_name]
        sensitive_attrs_to_cov_thresh = {sensitive_name: cov}

        gamma = None

        # print(np.array(y_train).shape)
        #y_train2 = [y[0] for y in self.y_train]
        model = ut.train_model(self.x_train, self.y_train, sensitive, loss_function, 1, 0, sep_constraint, sensitive_attrs,
                                    sensitive_attrs_to_cov_thresh, gamma)
        # self.model.fit(x_train, y_train)
        # scores_model(self.model.predict(x_test), y_test)
        pred_ = []
        for x in self.x_test:
            pred_.append(np.sign(np.dot(model, x)))
        scores_metrics(pred_, self.y_test)
        #y_pred = [[y] for y in pred_]
        fairness_metrics(self.x_test, self.y_test, pred_, self.sensitive_index, protected=self.protected,
                         unprotected=self.unprotected)
        return model


class CASE1:
    def __init__(self, data,target_index=None, sensitive_name='sex', sensitive_index=8):
        self.data = data
        self.data_init(data)
        self.sensitive_name = sensitive_name
        self.sensitive_index = sensitive_index
        self.target_index = target_index

        print('Sensitive attribute: ', sensitive_name)
    def data_init(self, data):
        self.train, self.test = data_split(data=data, sample_size=0.25)
        self.x_train, self.y_train = split_features_target(self.train, index=target_index)
        self.x_test, self.y_test = split_features_target(self.test, index=target_index)
    def most_correlated(self, data):
        model = DecisionTreeRegressor()
        other_features, sensitive = split_features_target(data, index=self.sensitive_index)
        # fit the model
        model.fit(other_features, sensitive)
        importance = model.feature_importances_
        most_correlated = 0
        most_correlated_index = 0
        for i, v in enumerate(importance):
            if v > most_correlated:
                most_correlated_index = i
        if most_correlated_index != self.target_index:
            data = np.delete(data, most_correlated, axis=1)
            self.target_index = self.target_index-1
            self.sensitive_index = self.sensitive_index-1
        return data

    def apply_sampling_BorderlineSMOTE(self, check_correlated=False):
        X, Y = split_features_target(self.data, index=self.target_index)
        X_sampled, Y_sampled = BorderlineSMOTE(random_state=42).fit_resample(X, Y)

        print(X_sampled.shape, Y_sampled.shape)
        #data_sampled = np.concatenate((X_sampled, Y_sampled), axis=0)
        data_sampled = np.column_stack((X_sampled, Y_sampled))
        if check_correlated:
            data_sampled = self.most_correlated(data_sampled)
        self.train, self.test = data_split(data=data_sampled, sample_size=0.25)
        self.x_train, self.y_train = split_features_target(self.train, index=self.target_index)
        self.x_test, self.y_test = split_features_target(self.test, index=self.target_index)
    def apply_sampling_SMOTE(self, check_correlated=False):
        X, Y = split_features_target(self.data, index=self.target_index)
        X_sampled, Y_sampled = SMOTE(random_state=42).fit_resample(X, Y)
        data_sampled = np.concatenate((X_sampled, Y_sampled), axis=1)
        if check_correlated:
            data_sampled = self.most_correlated(data_sampled)
        self.train, self.test = data_split(data=data_sampled, sample_size=0.25)
        self.x_train, self.y_train = split_features_target(self.train, index=target_index)
        self.x_test, self.y_test = split_features_target(self.test, index=target_index)
    def apply_sampling_RandomUnderSampler(self, check_correlated=False):
        X, Y = split_features_target(self.data, index=self.target_index)
        X_sampled, Y_sampled = RandomUnderSampler(random_state=42).fit_resample(X, Y)
        data_sampled = np.concatenate((X_sampled, Y_sampled), axis=1)
        if check_correlated:
            data_sampled = self.most_correlated(data_sampled)
        self.train, self.test = data_split(data=data_sampled, sample_size=0.25)
        self.x_train, self.y_train = split_features_target(self.train, index=target_index)
        self.x_test, self.y_test = split_features_target(self.test, index=target_index)
    def apply_sampling_EditedNearestNeighbours(self, check_correlated=False):
        X, Y = split_features_target(self.data, index=self.target_index)
        X_sampled, Y_sampled = EditedNearestNeighbours().fit_resample(X, Y)
        data_sampled = np.concatenate((X_sampled, Y_sampled), axis=1)
        if check_correlated:
            data_sampled = self.most_correlated(data_sampled)
        self.train, self.test = data_split(data=data_sampled, sample_size=0.25)
        self.x_train, self.y_train = split_features_target(self.train, index=target_index)
        self.x_test, self.y_test = split_features_target(self.test, index=target_index)
    def apply_sampling_SMOTEENN(self, check_correlated=False):
        X, Y = split_features_target(self.data, index=self.target_index)
        X_sampled, Y_sampled = SMOTEENN(random_state=42).fit_resample(X, Y)
        data_sampled = np.concatenate((X_sampled, Y_sampled), axis=1)
        if check_correlated:
            data_sampled = self.most_correlated(data_sampled)
        self.train, self.test = data_split(data=data_sampled, sample_size=0.25)
        self.x_train, self.y_train = split_features_target(self.train, index=target_index)
        self.x_test, self.y_test = split_features_target(self.test, index=target_index)

    def casuality(self, model_class, model, system_type='All'):
        x_train, y_train = model_class.apply_casuality_train(model, system_type=system_type)
        x_test, y_test = model_class.apply_casuality_test(model, system_type=system_type)
        train_combined = np.column_stack((x_train, y_train))
        test_combined = np.column_stack((x_test, y_test))

        data = np.append(train_combined, test_combined, axis=0)
        train, test = data_split(data=data, sample_size=0.25)
        x_train, y_train = split_features_target(train, index=self.target_index)
        x_test, y_test = split_features_target(test, index=self.target_index)

        return x_train, y_train, x_test, y_test

    def test_adult_data(self, apply_casuality=False):
        model_class = ModelClass(self.x_train, self.y_train, self.x_test, self.y_test,self.sensitive_index)
        #fit GaussianNB
        #model = model_class.fit_GaussianNB()
        '''if apply_casuality:
            #TODO: Apply casuality here
            print(' ******* Applying casuality on GaussianNB')
            x_train, y_train, x_test, y_test = self.casuality(model_class,model)
            model_class = ModelClass(x_train, y_train, x_test, y_test, self.sensitive_index)
            model_class.fit_GaussianNB()'''

        # fit DecisionTree
        model = model_class.fit_DecisionTree()
        if apply_casuality:
            #TODO: Apply casuality here
            print(' ******* Applying casuality on DecisionTree')
            x_train, y_train, x_test, y_test = self.casuality(model_class,model)
            model_class = ModelClass(x_train, y_train, x_test, y_test, self.sensitive_index)
            model_class.fit_DecisionTree()
        # fit RandomForest
        #model = model_class.fit_RandomForest()
        '''if apply_casuality:
            # TODO: Apply casuality here
            print(' ******* Applying casuality on RandomForest')
            x_train, y_train, x_test, y_test = self.casuality(model_class, model)
            model_class = ModelClass(x_train, y_train, x_test, y_test, self.sensitive_index)
            model_class.fit_RandomForest()'''
        # fit FairClassifier
        #model = model_class.fit_LogisticRegression()
        '''if apply_casuality:
            # TODO: Apply casuality here
            print(' ******* Applying casuality on RandomForest')
            x_train, y_train, x_test, y_test = self.casuality(model_class, model)
            model_class = ModelClass(x_train, y_train, x_test, y_test, self.sensitive_index)
            model_class.fit_LogisticRegression()'''
        # fit FairClassifier
        #model = model_class.fit_FairClassifier(self.sensitive_name, self.sensitive_index)
        '''if apply_casuality:
            # TODO: Apply casuality here
            print(' ******* Applying casuality on RandomForest')
            x_train, y_train, x_test, y_test = self.casuality(model_class, model,system_type='A')
            model_class = ModelClass(x_train, y_train, x_test, y_test, self.sensitive_index)
            model_class.fit_FairClassifier(self.sensitive_name, self.sensitive_index)'''
        #filename = 'adult.data'
        #sensitive_index = 8
        #columns = df_adult.columns.tolist()
        #columns.remove(target_column)
        #columns.remove('Probability')
        #print(columns)
        #df_X = df_adult[columns]
        #df_Y = df_adult[[sensitive_list[0]]]
        #df_Y = df_adult[[target_column]]
        #other_features, sensitive = split_features_target(df.to_numpy(), index=sensitive_index)
        #print(df_X.head())
        #print(df_Y.head())



if __name__ == '__main__':
    path = '../dataset/'
    ### Adult dataset
    target_column = 'Probability'
    loadData = LoadData(path)
    df_adult = loadData.load_adult_data('adult.data.csv')
    sensitive_list = loadData.sensitive_list
    sensitive_indices = loadData.sensitive_indices

    target_name = loadData.target_name
    target_index = loadData.target_index
    case1 = CASE1(df_adult.to_numpy(), target_index=target_index, sensitive_name=sensitive_list[1], sensitive_index=sensitive_indices[1])
    ## Apply sampling here
    case1.apply_sampling_BorderlineSMOTE(check_correlated=True)
    case1.test_adult_data(apply_casuality=True)
    #case1.test_adult_data(df_adult,target_column)

