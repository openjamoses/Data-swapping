from src.models.v3.load_data import LoadData


def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            print(corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                #if colname in dataset.columns:
                    #del dataset[colname] # deleting the column from the dataset
    return col_corr


path = '../dataset/'
### Adult dataset
#target_column = 'Probability'
loadData = LoadData(path)
data_name = 'student'
#df_adult = loadData.load_adult_data('adult.data.csv')
#df_adult = loadData.load_credit_defaulter('default_of_credit_card_clients.csv')
#df_adult = loadData.load_student_data('Student.csv')
df_adult = loadData.load_student_data('Student.csv')

col_corr = correlation(df_adult, 0.45)

print(col_corr)

