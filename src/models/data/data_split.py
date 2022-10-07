import xgboost
import shap
from pathlib import Path
from src.models.v3.load_data import LoadData
from src.models.v3.utility_functions import *
def fit_xgboost(X_train, y_train, X_test, y_test):
    """ Train an XGBoost model with early stopping.
    """
    dtrain = xgboost.DMatrix(X_train, label=y_train)
    dtest = xgboost.DMatrix(X_test, label=y_test)
    model = xgboost.train(
        { "eta": 0.001, "subsample": 0.5, "max_depth": 2, "objective": "reg:logistic"}, dtrain, num_boost_round=200000,
        evals=((dtest, "test"),),  verbose_eval=False
    )
    return model
path = '../dataset/'
### Adult dataset
#target_column = 'Probability'
correlation_threshold = 0.45 #0.35
loadData = LoadData(path,threshold=correlation_threshold) #,threshold=correlation_threshold
data_name = 'student-45_2_50' #_35_threshold

df_adult = loadData.load_adult_data('adult.data.csv')
#df_adult = loadData.load_credit_defaulter('default_of_credit_card_clients.csv')
#df_adult = loadData.load_student_data('Student.csv')
#df_adult = loadData.load_student_data('Student.csv')
sensitive_list = loadData.sensitive_list
sensitive_indices = loadData.sensitive_indices
colums_list = df_adult.columns.tolist()
target_index = loadData.target_index

data = df_adult.to_numpy()
train, test = data_split(data=data, sample_size=0.25)
train, val = data_split(data=data, sample_size=0.20)
path_data = '../dataset/experiments/data/'
Path(path_data).mkdir(parents=True, exist_ok=True)
#np.savetxt(path_data+"train_adult.csv", train, delimiter=",")
#np.savetxt(path_data+"test_adult.csv", test, delimiter=",")
#np.savetxt(path_data+"val_adult.csv", val, delimiter=",")
print(train.shape, test.shape, val.shape)
x_train, y_train = split_features_target(train,index=target_index)
x_test, y_test = split_features_target(test,index=target_index)

model = fit_xgboost(x_train,y_train, x_test, y_test)
from sklearn.model_selection import train_test_split

train, test = train_test_split(df_adult, test_size=0.25)
train, val = train_test_split(train, test_size=0.2)
dataset = 'adult'
Path(path_data+dataset).mkdir(parents=True, exist_ok=True)
train.to_csv(path_data+dataset+"/train.csv", index=False)
test.to_csv(path_data+dataset+"/test.csv", index=False)
val.to_csv(path_data+dataset+"/val.csv", index=False)
#default_of_credit_card_clients
dataset = 'default_of_credit_card_clients'
df_adult = loadData.load_credit_defaulter('default_of_credit_card_clients.csv')
train, test = train_test_split(df_adult, test_size=0.25)
train, val = train_test_split(train, test_size=0.2)

Path(path_data+dataset).mkdir(parents=True, exist_ok=True)
train.to_csv(path_data+dataset+"/train.csv", index=False)
test.to_csv(path_data+dataset+"/test.csv", index=False)
val.to_csv(path_data+dataset+"/val.csv", index=False)

dataset = 'Student'
df_adult = loadData.load_student_data('Student.csv')
train, test = train_test_split(df_adult, test_size=0.25)
train, val = train_test_split(train, test_size=0.2)
Path(path_data+dataset).mkdir(parents=True, exist_ok=True)
train.to_csv(path_data+dataset+"/train.csv", index=False)
test.to_csv(path_data+dataset+"/test.csv", index=False)
val.to_csv(path_data+dataset+"/val.csv", index=False)


dataset = 'GermanData'
df_adult = loadData.load_german_data('GermanData.csv')
train, test = train_test_split(df_adult, test_size=0.25)
train, val = train_test_split(train, test_size=0.2)
Path(path_data+dataset).mkdir(parents=True, exist_ok=True)
train.to_csv(path_data+dataset+"/train.csv", index=False)
test.to_csv(path_data+dataset+"/test.csv", index=False)
val.to_csv(path_data+dataset+"/val.csv", index=False)


dataset = 'compas'
#df = pd.read_csv(path+'compas-scores-two-years.csv')
#print(df)
df_adult = loadData.load_compas_data('compas-scores-two-years.csv')
print(dataset, df_adult.to_numpy())
train, test = train_test_split(df_adult, test_size=0.25)
train, val = train_test_split(train, test_size=0.2)
Path(path_data+dataset).mkdir(parents=True, exist_ok=True)
train.to_csv(path_data+dataset+"/train.csv", index=False)
test.to_csv(path_data+dataset+"/test.csv", index=False)
val.to_csv(path_data+dataset+"/val.csv", index=False)


dataset = 'clevelan_heart'
df_adult = loadData.load_clevelan_heart_data('processed.cleveland.data.csv')
train, test = train_test_split(df_adult, test_size=0.25)
train, val = train_test_split(train, test_size=0.2)
Path(path_data+dataset).mkdir(parents=True, exist_ok=True)
train.to_csv(path_data+dataset+"/train.csv", index=False)
test.to_csv(path_data+dataset+"/test.csv", index=False)
val.to_csv(path_data+dataset+"/val.csv", index=False)


dataset = 'bank'
df_adult = loadData.load_bank_data('bank.csv')
target_index = loadData.target_index
train, test = train_test_split(df_adult, test_size=0.25)
train, val = train_test_split(train, test_size=0.2)
Path(path_data+dataset).mkdir(parents=True, exist_ok=True)
train.to_csv(path_data+dataset+"/train.csv", index=False)
test.to_csv(path_data+dataset+"/test.csv", index=False)
val.to_csv(path_data+dataset+"/val.csv", index=False)

