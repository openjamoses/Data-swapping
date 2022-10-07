import random
import sys

from sklearn.naive_bayes import GaussianNB
from src.models.v2.utils import *
class TestSuit:
    def __init__(self, path='../dataset/', model=None, protected=0, sensitive_index=8):
        self.data = load_data_xlsx(path + 'Dataset.xlsx', sheet_name='Credit')
        self.model = model
        self.train, self.test = data_split(self.data)
        self.protected = protected
        self.sensitive_index = sensitive_index

        self.list_test_suites_str = []
        self.list_test_suites = []
        self.list_test_suites_new = [x for x in self.test]
        for i in range(len(self.test)):
            new_str = [str(v) for v in self.test[i]]
            new_str = ','.join(new_str)
            self.list_test_suites_str.append(new_str)

        self.fit()
    def fit(self):
        x_train, y_train = split_features_target(self.train)
        x_test, y_test = split_features_target(self.test)
        self.model.fit(x_train, y_train)
        scores_model(self.model.predict(x_test), y_test)

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
            prosperity_score, casuality_data = compute_prosperity_score(model=self.model, test_data=self.list_test_suites_new, sensitive=[0, 1],
                                                                        sensitive_index=self.sensitive_index)
            for x in casuality_data[CASUALITY_KEY]:
                self.list_test_suites.append(x)

            # print(casuality_data[CASUALITY_KEY])
            good_group = random.choices(casuality_data[CASUALITY_KEY], k=int(len(casuality_data[CASUALITY_KEY]) / 2))
            bad_group = random.choices(casuality_data[NON_CASUALITY_KEY], k=int(len(casuality_data[NON_CASUALITY_KEY]) / 2))

            combined_choices = []
            combined_choices.extend(good_group)
            combined_choices.extend(bad_group)

            #List_children = []
            for i in range(20):
                compute_disparate_subgroup_v2(self.model, combined_choices, self.list_test_suites_str, self.list_test_suites,system_type='C', sensitive_index=self.sensitive_index)
            for x in self.list_test_suites:
                self.list_test_suites_new.append(x)

    def export_data(self):
        np.savetxt(path_output+"test_suites_expanded_C8.csv", self.list_test_suites_new, delimiter=",")
        np.savetxt(path_output + "test_suites_casuality_C8.csv", self.list_test_suites, delimiter=",")

model = GaussianNB()
path = '../dataset/'
path_output = '../dataset/suites/'
if __name__ == '__main__':
    testsuit = TestSuit(path=path,model=model, protected=0,sensitive_index=8)
    testsuit.generate_testcases()
    testsuit.export_data()




