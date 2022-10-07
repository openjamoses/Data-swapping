import xgboost
import shap
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
X, Y = split_features_target(data, index=target_index)

train, test = data_split(data=data, sample_size=0.25)
x_train, y_train = split_features_target(train, index=target_index)
x_test, y_test = split_features_target(test, index=target_index)


model = fit_xgboost(x_train, y_train, x_test, y_test)

#explainer = shap.PermutationExplainer(model.predict, X)
#shap_values = explainer(X)

#clust = shap.utils.hclust(X, Y, linkage="single")
#shap.plots.bar(shap_values, clustering=clust, clustering_cutoff=1)

explainer = shap.PermutationExplainer(model.predict, x_test[:10,:])

# explain the first 10 predictions
# explaining each prediction requires 2 * background dataset size runs
# shap_values = explainer.shap_values(self.x_test)
shap_values = explainer(x_test[:5,:], max_evals=10 * x_test.shape[1], main_effects=False)

print('shap_values.data:', shap_values.data)
print('shap_values.values: ', shap_values.values)
print('shap_values.base_values: ', shap_values.base_values)
print('shap_values.feature_names: ', shap_values.feature_names)
#print('shap_values.base_values: ', shap_values.)
