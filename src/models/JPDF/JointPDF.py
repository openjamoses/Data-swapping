import numpy as np
import pandas as pd
import math
def variance(data, ddof=0):
    n = len(data)
    mean = sum(data) / n
    return sum((x - mean) ** 2 for x in data) / (n - ddof)
def stdev(data):
    var = variance(data)
    std_dev = math.sqrt(var)
    return std_dev


class JointPDF:
    def joint_pdf_one_discrete_one_continous(self, data, discrete_feature_name, continous_feature_name):
        Y = data[discrete_feature_name].values.tolist() # Disrete data
        X = data[continous_feature_name].values.tolist() # Continous data
        Ps = {}
        for i in range(len(Y)):
            if Y[i] in Ps.keys():
                Ps[Y[i]].append(X[i])
            else:
                Ps[Y[i]] = [X[i]]
        Mean_s = {}
        Mean_s2 = {}
        Variance_s = {}
        Stdev_s2 = {}
        for key, val in Ps.items():
            Mean_s[key] = np.sum(val)/len(val)
            Variance_s[key] = np.sum([i**2 for i in val])/len(val)-1
            Stdev_s2[key] = np.std(val)

            Mean_s2[key] = np.mean(val)

            print('Mean: {}'.format(key), Mean_s.get(key), 'Variance: {}'.format(key), Variance_s.get(key))
            print('Mean: {}'.format(key), Mean_s.get(key), 'Variance: {}'.format(key), variance(val), 'Std: ', stdev(val))
            print('Mean: {}'.format(key), Mean_s.get(key), 'Variance: {}'.format(key), np.std(val)**2, 'Std: ',
                  np.std(val))

        for x in set(X):
            sex_set = list(set(Y))
            val = Ps.get(sex_set[0])
            val2 = Ps.get(sex_set[1])

            p = len(val) / len(Y)
            q = len(val2) / len(Y)

            jpf = (p/(np.std(val)*math.sqrt(2*math.pi))) *np.exp(-1*(x-np.mean(val))**2/(2*np.std(val)))

            Pxy = pow((len(val) / len(Y)) / np.std(val) * math.sqrt(2 * math.pi),
                      -1 * (((X[i] - np.mean(val)) ** 2) / (2 * (np.std(val) ** 2)))) + pow(
                (len(val) - 1 / len(Y)) / np.std(val2) * math.sqrt(2 * math.pi),
                -1 * (((X[i] - np.mean(val2)) ** 2) / (2 * (np.std(val2) ** 2))))

        for i in range(len(X)):
            sex_set = list(set(Y))
            val = Ps.get(sex_set[0])
            val2 = Ps.get(sex_set[1])
            Pxy = pow((len(val)/len(Y))/np.std(val) *math.sqrt(2*math.pi), -1*(((X[i]-np.mean(val))**2)/(2*(np.std(val)**2)))) + pow((len(val)-1 / len(Y)) / np.std(val2) * math.sqrt(2 * math.pi),
                -1 * (((X[i] - np.mean(val2)) ** 2) / (2 * (np.std(val2) ** 2))))

            print(i, X[i], 'Pxy: ', Pxy)



if __name__ == '__main__':


    path = '../dataset/'
    df = pd.read_csv(path + 'adult.data.csv')

    jointPDF = JointPDF()
    jointPDF.joint_pdf_one_discrete_one_continous(df, 'sex', 'capital-gain')