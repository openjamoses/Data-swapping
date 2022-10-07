# Preparando dados iris
import math

import pandas as pd

path = '/Volumes/Cisco/Summer2022/Faireness/Experiments/FairTest-Experiments/src/models/dataset/Knn_hellinger_distance_iris_dataset-master/'
iris = pd.read_csv(path+'iris2.csv')
pd.set_option('mode.chained_assignment', None)

medias = pd.DataFrame()
medias['sepal_length'] = iris['sepal_length']
medias['sepal_width'] = iris['sepal_width']
medias['petal_length'] = iris['petal_length']
medias['petal_width'] = iris['petal_width']
medias['species'] = iris['species']

for i in range(150):
    if (iris['species'].iloc[i] == 'Iris-setosa'):
        medias['species'].iloc[i] = 1
    elif (iris['species'].iloc[i] == 'Iris-versicolor'):
        medias['species'].iloc[i] = 2
    elif (iris['species'].iloc[i] == 'Iris-virginica'):
        medias['species'].iloc[i] = 3

desvios = pd.DataFrame()
desvios['sepal_length'] = iris['sepal_length']
desvios['sepal_width'] = iris['sepal_width']
desvios['petal_length'] = iris['petal_length']
desvios['petal_width'] = iris['petal_width']
desvios['species'] = iris['species']

for i in range(150):
    if (iris['species'].iloc[i] == 'Iris-setosa'):
        desvios['species'].iloc[i] = 1
    elif (iris['species'].iloc[i] == 'Iris-versicolor'):
        desvios['species'].iloc[i] = 2
    elif (iris['species'].iloc[i] == 'Iris-virginica'):
        desvios['species'].iloc[i] = 3

medias_teste = pd.DataFrame()
medias_teste['sepal_length'] = iris['sepal_length']
medias_teste['sepal_width'] = iris['sepal_width']
medias_teste['petal_length'] = iris['petal_length']
medias_teste['petal_width'] = iris['petal_width']
medias_teste['species'] = iris['species']

for i in range(150):
    if (iris['species'].iloc[i] == 'Iris-setosa'):
        medias_teste['species'].iloc[i] = 1
    elif (iris['species'].iloc[i] == 'Iris-versicolor'):
        medias_teste['species'].iloc[i] = 2
    elif (iris['species'].iloc[i] == 'Iris-virginica'):
        medias_teste['species'].iloc[i] = 3

desvios_teste = pd.DataFrame()
desvios_teste['sepal_length'] = iris['sepal_length']
desvios_teste['sepal_width'] = iris['sepal_width']
desvios_teste['petal_length'] = iris['petal_length']
desvios_teste['petal_width'] = iris['petal_width']
desvios_teste['species'] = iris['species']

for i in range(150):
    if (iris['species'].iloc[i] == 'Iris-setosa'):
        desvios_teste['species'].iloc[i] = 1
    elif (iris['species'].iloc[i] == 'Iris-versicolor'):
        desvios_teste['species'].iloc[i] = 2
    elif (iris['species'].iloc[i] == 'Iris-virginica'):
        desvios_teste['species'].iloc[i] = 3
# Gerando os desvios
import numpy as np

for j in range(150):
    desvios.sepal_length[j] = np.random.normal(0, (desvios.sepal_length[j] * 10 / 100))
    desvios.sepal_length[j] = round(desvios.sepal_length[j], 2)
    desvios.sepal_width[j] = np.random.normal(0, (desvios.sepal_width[j] * 10 / 100))
    desvios.sepal_width[j] = round(desvios.sepal_width[j], 2)
    desvios.petal_length[j] = np.random.normal(0, (desvios.petal_length[j] * 15 / 100))
    desvios.petal_length[j] = round(desvios.petal_length[j], 2)
    desvios.petal_width[j] = np.random.normal(0, (desvios.petal_width[j] * 60 / 100))
    desvios.petal_width[j] = round(desvios.petal_width[j], 2)

for j in range(150):
    desvios_teste.sepal_length[j] = -1
    desvios_teste.sepal_width[j] = -1
    desvios_teste.petal_length[j] = -1
    desvios_teste.petal_width[j] = -1

    medias_teste.sepal_length[j] = -1
    medias_teste.sepal_width[j] = -1
    medias_teste.petal_length[j] = -1
    medias_teste.petal_width[j] = -1
# Separando 10 amostras de cada flor para teste/validação

for j in range(0, 10, 1):
    medias_teste.sepal_length[j] = medias.sepal_length[j]
    medias_teste.sepal_width[j] = medias.sepal_width[j]
    medias_teste.petal_length[j] = medias.petal_length[j]
    medias_teste.petal_width[j] = medias.petal_width[j]
    medias_teste.species[j] = medias.species[j]

    desvios_teste.sepal_length[j] = desvios.sepal_length[j]
    desvios_teste.sepal_width[j] = desvios.sepal_width[j]
    desvios_teste.petal_length[j] = desvios.petal_length[j]
    desvios_teste.petal_width[j] = desvios.petal_width[j]
    desvios_teste.species[j] = desvios.species[j]

j = j + 1
for k in range(50, 60, 1):
    medias_teste.sepal_length[j] = medias.sepal_length[k]
    medias_teste.sepal_width[j] = medias.sepal_width[k]
    medias_teste.petal_length[j] = medias.petal_length[k]
    medias_teste.petal_width[j] = medias.petal_width[k]
    medias_teste.species[j] = medias.species[k]

    desvios_teste.sepal_length[j] = desvios.sepal_length[k]
    desvios_teste.sepal_width[j] = desvios.sepal_width[k]
    desvios_teste.petal_length[j] = desvios.petal_length[k]
    desvios_teste.petal_width[j] = desvios.petal_width[k]
    desvios_teste.species[j] = desvios.species[k]

    j = j + 1

for k in range(140, 150, 1):
    medias_teste.sepal_length[j] = medias.sepal_length[k]
    medias_teste.sepal_width[j] = medias.sepal_width[k]
    medias_teste.petal_length[j] = medias.petal_length[k]
    medias_teste.petal_width[j] = medias.petal_width[k]
    medias_teste.species[j] = medias.species[k]

    desvios_teste.sepal_length[j] = desvios.sepal_length[k]
    desvios_teste.sepal_width[j] = desvios.sepal_width[k]
    desvios_teste.petal_length[j] = desvios.petal_length[k]
    desvios_teste.petal_width[j] = desvios.petal_width[k]
    medias_teste.species[j] = medias.species[k]
    j = j + 1
medias_teste = medias_teste.drop(medias_teste[(medias_teste.sepal_width == -1)].index)
desvios_teste = desvios_teste.drop(desvios_teste[(desvios_teste.sepal_width == -1)].index)
print(medias_teste.shape, desvios_teste.shape)

for i in range(0, 10, 1):
    medias.drop(index=i, inplace=True)
    desvios.drop(index=i, inplace=True)

for i in range(50, 60, 1):
    medias.drop(index=i, inplace=True)
    desvios.drop(index=i, inplace=True)

for i in range(140, 150, 1):
    medias.drop(index=i, inplace=True)
    desvios.drop(index=i, inplace=True)


# Dados ajustados

print("Dados treino")
print(medias.shape)
print(desvios.shape)


print("\nDados teste")
print(medias_teste.shape)
print(desvios_teste.shape)

#print(medias_teste)

## organizando os index
medias = medias.set_index([[x for x in range(120)], 'sepal_length','sepal_width','petal_length', 'petal_width','species'])
desvios = desvios.set_index([[x for x in range(120)], 'sepal_length','sepal_width','petal_length', 'petal_width','species'])

medias_teste = medias_teste.set_index([[x for x in range(30)], 'sepal_length','sepal_width','petal_length', 'petal_width','species'])
desvios_teste = desvios_teste.set_index([[x for x in range(30)], 'sepal_length','sepal_width','petal_length', 'petal_width','species'])


def hellinger(ponto_a, ponto_b):
    epsilon = 1e-6
    if ponto_a[1] == 0:
        ponto_a = (ponto_a[0], epsilon)
    elif ponto_b[1] == 0:
        ponto_b = (ponto_b[0], epsilon)

    if ponto_a[1] < 0:
        ponto_a = (ponto_a[0], ponto_a[1] * -1)
    if ponto_b[1] < 0:
        ponto_b = (ponto_b[0], ponto_b[1] * -1)

    expoente = -0.25 * (ponto_a[0] - ponto_b[0]) ** 2 / (ponto_a[1] ** 2 + ponto_b[1] ** 2)
    raiz = math.sqrt(2 * ponto_a[1] * ponto_b[1] / (ponto_a[1] ** 2 + ponto_b[1] ** 2))
    distance = math.sqrt(1 - (raiz * math.exp(expoente)))

    return distance


def hellinger_multi(ponto):
    distance = 0
    d = 0
    distances = []
    for i in range(120):
        distance = 0
        for k in range(4):
            ponto_a = (ponto[0][k], ponto[1][k])
            ponto_b = (medias.iloc[i].name[k + 1], desvios.iloc[i].name[k + 1])
            # print(ponto_a)
            # print(medias.iloc[i].name)
            d = hellinger(ponto_a, ponto_b)

            distance += d
        distances.append(distance)

    return distances

# get distances of test_row vs all training rows


data_= (medias_teste.iloc[i].name[1:],desvios_teste.iloc[i].name[1:])
print(data_)
distancias = hellinger_multi(data_)

print('distancias: ', distancias)
# Sort by distance and return indices of the first k neighbors
k_idx = np.argsort(distancias)[: 3]
print('k_idx: ', k_idx)
