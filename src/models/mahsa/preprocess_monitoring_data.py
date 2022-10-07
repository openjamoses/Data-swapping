import csv

import pandas as pd
import numpy as np
path = '/Volumes/Cisco/Summer2022/Mahsa/new monitoring/'

files = ['h_5_5_1w_2.csv', 'h_5_5_2w_1c.csv', 'h_5_5_pythonscript_2.csv']
data_file = open(path+'combined_monitoring_data.csv', mode='w', newline='',
                      encoding='utf-8')
data_writer = csv.writer(data_file)
data_writer.writerow(
    ['row_index','time', 'cpu_percentage', 'memory', 'sensors', 'workers'])

for file_ in files:
    split_name = file_.split('_')
    data = pd.read_csv(path+file_)
    grouped_cpu = {}
    grouped_memory = {}
    discrete_time = data.discrete_time.values.tolist()
    cpu_percent = data.cpu_percent.values.tolist()
    memory = data.memory.values.tolist()
    for i in range(len(data)):
        if discrete_time[i] in grouped_cpu.keys():
            grouped_cpu[discrete_time[i]].append(cpu_percent[i])
            grouped_memory[discrete_time[i]].append(memory[i])
        else:
            grouped_cpu[discrete_time[i]] = [cpu_percent[i]]
            grouped_memory[discrete_time[i]] = [memory[i]]
    index = 0
    for key, val in grouped_cpu.items():
        data_writer.writerow(
            [index, key, np.sum(val), np.sum(grouped_memory.get(key)), split_name[1], split_name[3]])
        index += 1

data_file.close()
