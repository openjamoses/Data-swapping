import pandas as pd
import numpy as np
import csv

from src.models.v3.load_data import LoadData

path = '../../dataset/'

correlation_threshold = 0.45 #0.35
loadData = LoadData(path,threshold=correlation_threshold) #,threshold=correlation_threshold
data_name = 'adult-45' #_35_threshold

df_adult = loadData.load_adult_data('adult.data.csv')
#df_adult = loadData.load_credit_defaulter('default_of_credit_card_clients.csv')
#df_adult = loadData.load_student_data('Student.csv')
#df_adult = loadData.load_student_data('Student.csv')
sensitive_list = loadData.sensitive_list
sensitive_indices = loadData.sensitive_indices
colums_list = df_adult.columns.tolist()

slicing = 4
data = df_adult.to_numpy()
data_split = np.array_split(data, slicing)


print(data_split[0])
print(data_split[1])
print(data_split[2])
print(data_split[3])

rng = np.random.default_rng()
arr = np.arange(9).reshape((3, 3))
print(arr)
print(rng.shuffle(arr))