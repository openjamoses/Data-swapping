import csv
import os

import pandas as pd
import numpy as np
path = '/Volumes/Cisco/Summer2022/Mahsa/monitoring/'

folders = ['2W', '1w', 'pythonscript']
data_file = open(path+'combined_monitoring_data_overal.csv', mode='w', newline='',
                      encoding='utf-8')
data_writer = csv.writer(data_file)
data_writer.writerow(
    ['row_index','cpu_percentage', 'memory', 'sensors', 'workers'])

for folder in folders:
    file_list = os.listdir(path+folder)
    for file_ in file_list:
        print(path + folder+'/'+file_)
        split_name = file_.split('_')
        data = pd.read_csv(path + folder+'/'+file_)
        container_name = data.container_name.values.tolist()
        cpu_percent = data.cpu_percent.values.tolist()
        memory = data.memory.values.tolist()
        count = 0
        index = 0
        grouped_cpu = {}
        grouped_memory = {}
        for i in range(len(container_name)):
            if index in grouped_cpu.keys():
                grouped_cpu[index].append(cpu_percent[i])
                grouped_memory[index].append(memory[i])
            else:
                grouped_cpu[index] = [cpu_percent[i]]
                grouped_memory[index] = [memory[i]]
            count += 1
            if count == 4:
                count = 0
                index += 1
        for key, val in grouped_cpu.items():
            data_writer.writerow(
                [key, np.sum(val), np.sum(grouped_memory.get(key)), split_name[1]+' sensor(s)', folder])
data_file.close()
