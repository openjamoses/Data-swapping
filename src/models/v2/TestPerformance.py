import numpy as np
import pandas as pd

from src.models.v2.utils import *

path_input="../dataset/suites/"
#path_output="/content/drive/MyDrive/Fairness/Dataset/Tests/"
data = pd.read_csv(path_input + 'test_suites_casuality_A8.csv', header=None).to_numpy()
compute_disparate_impact(data, protected=0, sensitive_index=8)
data = pd.read_csv(path_input + 'test_suites_expanded_v8.csv', header=None).to_numpy()
compute_disparate_impact(data, protected=0, sensitive_index=8)

data = pd.read_csv(path_input + 'test_suites_casuality9.csv', header=None).to_numpy()
compute_disparate_impact(data, protected=0, sensitive_index=9)

data = pd.read_csv(path_input + 'test_suites_casuality_A8.csv', header=None).to_numpy()
compute_disparate_impact(data, protected=0, sensitive_index=8)
data = pd.read_csv(path_input + 'test_suites_casuality_A82.csv', header=None).to_numpy()
compute_disparate_impact(data, protected=0, sensitive_index=8)


data = pd.read_csv(path_input + 'test_suites_casuality_A92.csv', header=None).to_numpy()
compute_disparate_impact(data, protected=0, sensitive_index=9)


data = pd.read_csv(path_input + 'test_suites_casuality_C8.csv', header=None).to_numpy()
compute_disparate_impact(data, protected=0, sensitive_index=8)