import pandas as pd
import numpy as np

path = '../../dataset/logging/'

data = pd.read_csv(path+'log_kl_divergence_adult-45.csv')

Feature = data.Feature.values.tolist()
Category = data.Category.values.tolist()
Acc = data.Acc.values.tolist()
Acc_after = data.Acc_after.values.tolist()
KL_Pos = data.KL_Pos.values.tolist()
KL_Neg = data.KL_Neg.values.tolist()
Casuality = data.Casuality.values.tolist()
Importance = data.Importance.values.tolist()
SP = data.SP.values.tolist()
SP_after = data.SP_after.values.tolist()
TPR = data.TPR.values.tolist()
TPR_after = data.TPR_after.values.tolist()
FPR = data.FPR.values.tolist()
FPR_after = data.FPR_after.values.tolist()

data = data.groupby(['Feature','Category'])['Acc', 'Acc_after', 'KL_Pos', 'KL_Neg', 'Casuality', 'Importance', 'SP', 'SP_after', 'FPR', 'FPR_after'].mean()
data2 = data.groupby(['Feature'])['Acc', 'Acc_after', 'KL_Pos', 'KL_Neg', 'Casuality', 'Importance', 'SP', 'SP_after', 'FPR', 'FPR_after'].median()
data.to_csv(path+'exported_log_kl_divergence_adult-45.csv')
data2.to_csv(path+'exported_log_kl_divergence_adult-45_combined-median.csv')

print(data)

