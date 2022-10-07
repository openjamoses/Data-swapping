import pandas as pd
import numpy as np
import math

path = '../dataset/'
df = pd.read_csv(path+'adult.data.csv')
sex_data = df.sex.values.tolist()
age_data = df.age.values.tolist()

prob_sex = {}
mean_data = {}

for i in range(len(sex_data)):
    prob_sex[sex_data[i]] = prob_sex.get(sex_data[i], 0) +1
    if sex_data[i] in mean_data.keys():
        mean_data[sex_data[i]].append(age_data[i])
    else:
        mean_data[sex_data[i]] = [age_data[i]]
prob_sex2 = {}
for key, val in prob_sex.items():
    prob_sex2[key] = val/sum(list(prob_sex.values()))
    #todo: fx,y(x,y) = p/sqrt(2*math.pi) e**-(x-mean)**2
    p = val/sum(list(prob_sex.values()))


#print(prob_sex2)

fxy = 0
for i in range(len(age_data)):
    p = prob_sex2.get(sex_data[i])
    fxy_ = pow((p / np.var(age_data)*math.sqrt(2 * math.pi)), -1*(((age_data[i]-np.mean(mean_data.get(sex_data[i])))**2)/2*np.std(age_data)))
    print(i, fxy_)
    fxy += fxy_
print(fxy)