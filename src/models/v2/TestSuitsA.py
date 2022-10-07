import random
import sys

from sklearn.naive_bayes import GaussianNB
sys.path.insert(0, '../../examples/fair_classification/') # the code for fair classification is in this directory
import utils as ut
import numpy as np
import loss_funcs as lf # loss funcs that can be optimized subject to various constraints

from src.models.v2.utils import *
class TestSuit:
    def __init__(self, path='../dataset/', model=None, protected=0, sensitive_index=8):
        self.data = load_data_xlsx(path + 'Dataset.xlsx', sheet_name='Credit')
        self.model = model
        self.train, self.test = data_split(self.data, sample_size=0.2)
        self.protected = protected
        self.sensitive_index = sensitive_index

        if (self.sensitive_index == 9):
            self.name = 'sex'
            self.cov = 0
        else:
            self.name = 'race'
            self.cov = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

        self.list_test_suites_str = []
        self.list_test_suites = []
        self.list_test_suites_new = [x for x in self.test]
        for i in range(len(self.test)):
            new_str = [str(v) for v in self.test[i]]
            new_str = ','.join(new_str)
            self.list_test_suites_str.append(new_str)

        self.fitA()
    def fitA(self):
        x_train, y_train = split_features_target(self.train)
        x_test, y_test = split_features_target(self.test)
        self.sen = []
        for i in range(len(x_train)):
            self.sen.append(x_train[i][self.sensitive_index])
        sensitive = {}
        sens = []
        sensitive[self.name] = np.array(sens, dtype=float)
        loss_function = lf._logistic_loss
        sep_constraint = 0
        sensitive_attrs = [self.name]
        sensitive_attrs_to_cov_thresh = {self.name: self.cov}

        gamma = None

        #print(np.array(y_train).shape)
        y_train2 = [y[0] for y in y_train]
        self.model = ut.train_model(x_train, y_train2, sensitive, loss_function, 1, 0, sep_constraint, sensitive_attrs,
                           sensitive_attrs_to_cov_thresh, gamma)
        #self.model.fit(x_train, y_train)
        #scores_model(self.model.predict(x_test), y_test)


    def print_disparate(self):
        print('Disparate Impact Train dataset: ----------')
        compute_disparate_impact(self.train, protected=0, sensitive_index=self.sensitive_index)
        #compute_disparate_impact(self.train, protected=0, sensitive_index=9)
        print('*************************************************\n')

        print('Disparate Impact Test dataset: ----------')
        compute_disparate_impact(self.test, protected=0, sensitive_index=self.sensitive_index)
        #compute_disparate_impact(self.test, protected=0, sensitive_index=9)
        print('*************************************************\n')

        print('Disparate Impact Complete Data: ----------')
        compute_disparate_impact(self.data, protected=0, sensitive_index=self.sensitive_index)
        #compute_disparate_impact(self.data, protected=0, sensitive_index=9)
        print('*************************************************\n')
    def generate_testcases(self):
        for iterate_ in range(5):
            prosperity_score, casuality_data = compute_prosperity_score(model=self.model, test_data=self.list_test_suites_new, system_type='A', sensitive=[0, 1],
                                                                        sensitive_index=self.sensitive_index)
            if CASUALITY_KEY in casuality_data.keys():
                for x in casuality_data[CASUALITY_KEY]:
                    self.list_test_suites.append(x)

            # print(casuality_data[CASUALITY_KEY])
            if CASUALITY_KEY in casuality_data.keys():
                good_group = random.choices(casuality_data[CASUALITY_KEY], k=int(len(casuality_data[CASUALITY_KEY]) / 2))
            bad_group = random.choices(casuality_data[NON_CASUALITY_KEY], k=int(len(casuality_data[NON_CASUALITY_KEY]) / 2))

            combined_choices = []
            if CASUALITY_KEY in casuality_data.keys():
                combined_choices.extend(good_group)
            combined_choices.extend(bad_group)

            #List_children = []
            for i in range(20):
                compute_disparate_subgroup_v2(self.model, combined_choices, self.list_test_suites_str, self.list_test_suites,system_type='A', sensitive_index=self.sensitive_index)
            for x in self.list_test_suites:
                self.list_test_suites_new.append(x)

    def export_data(self):
        np.savetxt(path_output+"test_suites_expanded_A92.csv", self.list_test_suites_new, delimiter=",")
        np.savetxt(path_output + "test_suites_casuality_A92.csv", self.list_test_suites, delimiter=",")

## SystemA


#model = GaussianNB()
path = '../dataset/'
path_output = '../dataset/suites/'
if __name__ == '__main__':
    testsuit = TestSuit(path=path, protected=0,sensitive_index=9)
    testsuit.generate_testcases()
    testsuit.export_data()




