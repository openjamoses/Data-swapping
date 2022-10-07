# define the model
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from src.models.v3.load_data import LoadData
from src.models.v3.utility_functions import *

model = DecisionTreeClassifier()
class FeatureImportance:
    def __init__(self, data, target_index):
        self.data = data
        self.target_index = target_index
        self.data_init(data)


    def data_init(self, data):
        self.train, self.test = data_split(data=data, sample_size=0.25)
        self.x_train, self.y_train = split_features_target(self.train, index=self.target_index)
        self.x_test, self.y_test = split_features_target(self.test, index=self.target_index)

    def decision_tree(self, colums_list):
        # fit the model
        model.fit(self.x_train, self.y_train)
        # get importance
        importance = model.feature_importances_
        # summarize feature importance
        dict_importance = {}
        for i, v in enumerate(importance):
            dict_importance[colums_list[i]] = v
            print('Feature: %0d, %s, Score: %.5f' % (i,colums_list[i], v))
        # plot feature importance
        print(sort_dict(dict_importance, reverse=True))

        plt.bar([x for x in range(len(importance))], importance)
        plt.show()

if __name__ == '__main__':
    path = '../dataset/'
    ### Adult dataset
    # target_column = 'Probability'
    correlation_threshold = 0.45  # 0.35
    loadData = LoadData(path, threshold=correlation_threshold)  # ,threshold=correlation_threshold
    data_name = 'student-45_1_50'  # _35_threshold

    # df_adult = loadData.load_adult_data('adult.data.csv')
    # df_adult = loadData.load_credit_defaulter('default_of_credit_card_clients.csv')
    df_adult = loadData.load_student_data('Student.csv')
    # df_adult = loadData.load_student_data('Student.csv')
    sensitive_list = loadData.sensitive_list
    sensitive_indices = loadData.sensitive_indices
    colums_list = df_adult.columns.tolist()
    target_index = loadData.target_index

    fimportance = FeatureImportance(df_adult.to_numpy(), target_index)
    fimportance.decision_tree(colums_list)


