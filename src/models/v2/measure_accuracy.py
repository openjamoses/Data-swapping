import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from src.models.v2.utils import *
path1 = "../dataset/"
path_input="../dataset/suites/"
#path_output="/content/drive/MyDrive/Fairness/Dataset/Tests/"
data = load_data_xlsx(path1 + 'Dataset.xlsx', sheet_name='Credit')
train, test = data_split(data)
model = GaussianNB()

x_train, y_train = split_features_target(train)
x_test, y_test = split_features_target(test)
model.fit(x_train, y_train)
scores_model(model.predict(x_train), y_train)
scores_model(model.predict(x_test), y_test)


data2 = pd.read_csv(path_input + 'test_suites_casuality_A8.csv', header=None).to_numpy()
X,Y = split_features_target(data2)
scores_model(model.predict(X), Y)