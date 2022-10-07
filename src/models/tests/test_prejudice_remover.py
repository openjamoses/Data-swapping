from src.models.v3.load_data import LoadData

path = '../dataset/'
### Adult dataset
#target_column = 'Probability'
correlation_threshold = 0.35 #0.35
loadData = LoadData(path,threshold=correlation_threshold) #,threshold=correlation_threshold
data_name = 'adult-45' #_35_threshold

#df_adult = loadData.load_adult_data('adult.data.csv')
#df_adult = loadData.load_credit_defaulter('default_of_credit_card_clients.csv')
df_adult = loadData.load_student_data('Student.csv')
#df_adult = loadData.load_student_data('Student.csv')
sensitive_list = loadData.sensitive_list
sensitive_indices = loadData.sensitive_indices
colums_list = df_adult.columns.tolist()