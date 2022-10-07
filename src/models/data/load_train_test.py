import pandas as pd
class LoadTrainTest:
    def __init__(self, path):
        self.path = path
    def read_csv(self, filename):
        return pd.read_csv(self.path+filename+'/train.csv'), pd.read_csv(self.path+filename+'/test.csv') #.dropna()
    def load_adult(self, name_):
        train, test = self.read_csv(name_)
        self.sensitive_list = ['sex', 'race', 'age']
        self.sensitive_indices = [train.columns.tolist().index('sex'), train.columns.tolist().index('race'), train.columns.tolist().index('age')]
        self.target_name = 'Probability'
        self.target_index = train.columns.tolist().index(self.target_name)
        return train, test, self.target_index
    def load_compas(self, name_):
        train, test = self.read_csv(name_)
        self.sensitive_list = ['sex', 'race', 'age_cat']
        self.sensitive_indices = [train.columns.tolist().index('sex'), train.columns.tolist().index('race'), train.columns.tolist().index('age_cat')]
        self.target_name = 'Probability'
        self.target_index = train.columns.tolist().index(self.target_name)
        return train, test, self.target_index
    def load_german(self, name_):
        train, test = self.read_csv(name_)
        self.sensitive_list = ['sex']
        self.sensitive_indices = [train.columns.tolist().index('sex')]
        self.target_name = 'Probability'
        self.target_index = train.columns.tolist().index(self.target_name)
        return train, test, self.target_index
    def load_clevelan_heart(self, name_):
        train, test = self.read_csv(name_)
        self.sensitive_list = ['age']
        self.sensitive_indices = [train.columns.tolist().index('age')]
        self.target_name = 'Probability'
        self.target_index = train.columns.tolist().index(self.target_name)
        return train, test, self.target_index
    def load_bank(self, name_):
        train, test = self.read_csv(name_)
        self.sensitive_list = ['age']
        self.sensitive_indices = [train.columns.tolist().index('age')]
        self.target_name = 'Probability'
        self.target_index = train.columns.tolist().index(self.target_name)
        return train, test, self.target_index
    def load_credit_defaulter(self, name_):
        train, test = self.read_csv(name_)
        self.sensitive_list = ['SEX', 'AGE']
        self.sensitive_indices = [train.columns.tolist().index('SEX'), train.columns.tolist().index('AGE')]
        self.target_name = 'default payment next month'
        self.target_index = train.columns.tolist().index(self.target_name)
        return train, test, self.target_index
    def load_student(self, name_):
        train, test = self.read_csv(name_)
        self.sensitive_list = ['sex']
        self.sensitive_indices = [train.columns.tolist().index('sex')]
        self.target_name = 'Probability'
        self.target_index = train.columns.tolist().index(self.target_name)
        return train, test, self.target_index
