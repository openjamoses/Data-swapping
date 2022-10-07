import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn import metrics
CASUALITY_KEY = 'casuality'
NON_CASUALITY_KEY = 'noncasuality'
def load_data_csv(path):
    return pd.read_csv(path).to_numpy()
def load_data_xlsx(path, sheet_name='Credit'):
    return pd.read_excel(open(path, 'rb'), sheet_name=sheet_name, engine="openpyxl").to_numpy()
def split_features_target(data, index=20):
    return data[0:, 0:index], data[0:, index:]
def data_split(data, sample_size=0.2):
    return train_test_split(data, test_size=sample_size, random_state=42)
def scores_model(y_pred, y_test):
    accuracy = metrics.accuracy_score(y_test, y_pred)          # accuracy: (tp + tn) / (p + n)
    precision = metrics.precision_score(y_test, y_pred)        # precision tp / (tp + fp)
    recall = metrics.recall_score(y_test, y_pred)              # recall: tp / (tp + fn)
    f1 = metrics.f1_score(y_test, y_pred)                      # f1: 2 tp / (2 tp + fp + fn)

    print('Precision: ',precision, 'Recall: ', recall, 'F1: ', f1, 'Accuracy: ', accuracy)
    return precision, recall, f1, accuracy
def compute_disparate_impact(data, protected=0, sensitive_index=8):
    s0 = 0
    pos0 = 0

    s1 = 0
    pos1 = 0
    for i in range(len(data)):
        if data[i][sensitive_index] == protected:
            s0 += 1
            if data[i][-1] == 1:
                pos0 += 1
        else:
            s1 += 1
            if data[i][-1] == 1:
                pos1 += 1
    P0 = pos0/s0
    P1 = pos1/s1
    disparate_impact = round((P0/P1), 4)
    print('disparate_impact: ', disparate_impact)
    return disparate_impact
def compute_prosperity_score(model, test_data, system_type='C',sensitive=[0,1], sensitive_index=8):
    prosperity_count = 0
    casuality_data = {}
    casuality_data_index = {}
    for i in range(len(test_data)):
        row = []
        for j in range(len(test_data[i])):
            temp_val = test_data[i][j]
            if test_data[i][j] == sensitive[0] and j == sensitive_index:
                temp_val = sensitive[1]
            elif test_data[i][j] == sensitive[1] and j == sensitive_index:
                temp_val = sensitive[0]
            row.append(temp_val)
        row2 = [x for x in row[:-1]]
        if system_type == 'A':
            pred_original = np.sign(np.dot(model, test_data[i][:-1])) # model.predict()
            pred_after = np.sign(np.dot(model,row[:-1]))
        else:
            pred_original = model.predict(np.reshape(np.array(test_data[i][:-1]), [-1, 20]))
            pred_after = model.predict(np.reshape(np.array(row[:-1]), [-1, 20]))
        row2.append(pred_after)

        print(pred_original, pred_after)
        if pred_original != pred_after:
            prosperity_count += 1
            if CASUALITY_KEY in casuality_data.keys():
                #casuality_data[CASUALITY_KEY].append(test_data[i])
                casuality_data[CASUALITY_KEY].append(row2)
                casuality_data_index[CASUALITY_KEY].append(i)
            else:
                #casuality_data[CASUALITY_KEY] = [test_data[i]]
                casuality_data[CASUALITY_KEY] = [row2]
                casuality_data_index[CASUALITY_KEY] = [i]
        else:
            if NON_CASUALITY_KEY in casuality_data.keys():
                casuality_data[NON_CASUALITY_KEY].append(test_data[i])
                #casuality_data[NON_CASUALITY_KEY].append(row)
                casuality_data_index[NON_CASUALITY_KEY].append(i)
            else:
                casuality_data[NON_CASUALITY_KEY] = [test_data[i]]
                casuality_data_index[NON_CASUALITY_KEY] = [i]
    prosperity_score = round((prosperity_count/len(test_data)),4)
    print(casuality_data.keys())
    if CASUALITY_KEY in casuality_data.keys():
        print('Casuality: ', len(casuality_data[CASUALITY_KEY]), ' , None-Casuality: ', len(casuality_data[NON_CASUALITY_KEY]), ', Prosperity Score: ', prosperity_score)
    return prosperity_score, casuality_data
def compute_disparate_subgroup_v2(model, choice_list, list_children, list_test_suites, system_type='C', sensitive_index=8, epoches=200):
    for epoch in range(epoches):
        # Case1
        random_choice = random.choices(choice_list, k=2)

        #print('random_choice: ', random_choice)
        #for choice_index in random_choice:

        new_test_sub = []
        for ch_id in range(len(random_choice[0])):
            new_list1 = []
            new_list2 = []
            if ch_id != sensitive_index:
                ## Case 1
                val1 = random_choice[0][ch_id]
                val2 = random_choice[1][ch_id]
                for j in range(len(random_choice[0])):
                    if j == ch_id:
                        new_list1.append(val2)
                        new_list2.append(val1)
                    else:
                        new_list1.append(random_choice[0][j])
                        new_list2.append(random_choice[1][j])

            else:
                new_list1.append(random_choice[0][ch_id])
                new_list2.append(random_choice[1][ch_id])
            if len(new_list1)>2:
                new_test_sub.append(new_list1)
                new_test_sub.append(new_list2)

                ## Case 2
        random_choice2 = random.choices(choice_list, k=2)
        index_l = []
        for i in range(2,len(random_choice2[0])):
            if i != sensitive_index:
                index_l.append(i)
        for i in index_l:
            random_index = random.choices(random_choice2[0], k=i)
            L1 = []
            L2 = []
            for ch_id in range(len(random_choice2[0])):
                #if ch_id != sensitive_index:
                if ch_id in random_index:
                    L1.append(random_choice2[1][ch_id])
                    L2.append(random_choice2[0][ch_id])
                else:
                    L1.append(random_choice2[0][ch_id])
                    L2.append(random_choice2[1][ch_id])
            if len(new_list1) > 2:
                new_test_sub.append(L1)
                new_test_sub.append(L2)





        #for i in new_test_sub:
        #    print(i, len(i))
        new_test_sub = np.array(new_test_sub)

        list_index_new = []
        list_index_new2 = []
        for i in range(len(new_test_sub)):
            x2 = []
            #print(new_test_sub[i])
            for index in range(len(new_test_sub[i])):
                val = new_test_sub[i][index]
                #print(new_test_sub[i][index])
                if new_test_sub[i][index] == 1 and index == sensitive_index:
                    val = 0
                elif new_test_sub[i][index] == 0 and index == sensitive_index:
                    val = 1
                x2.append(val)
            x2 = np.array(x2)
            # print(new_x_test_sub[i], x2)
            if system_type == 'A':
                pred_x1 = np.sign(np.dot(model, new_test_sub[i][:20]))
                pred_x2 = np.sign(np.dot(model, x2[:20]))
            else:
                pred_x1 = model.predict(np.reshape(new_test_sub[i][:20], (-1, 20)))
                pred_x2 = model.predict(np.reshape(x2[:20], (-1, 20)))
            #if isinstance(pred_x1, )
            if pred_x1 != pred_x2: # change back to  pred_x1[0] != pred_x2[0]:
                # p_ += 1
                list_index_new.append(i)
            else:
                list_index_new2.append(i)
        new_str = [str(v) for v in new_test_sub[i]]
        new_str = ','.join(new_str)
        if not new_str in list_children:
            list_children.append(new_str)
            list_test_suites.append(new_test_sub[i])
        print('Children generated: ', len(list_children))

def compute_disparate_subgroup(model, choice_list, list_children, list_test_suites, system_type='C', sensitive_index=8, epoches=200):
    for epoch in range(epoches):
        # Case1
        random_choice = random.choices(choice_list, k=2)

        #print('random_choice: ', random_choice)
        test_a1 = list(random_choice[0][:sensitive_index])
        test_b1 = list(random_choice[1][:sensitive_index])
        for x in random_choice[1][sensitive_index:]:
            test_a1.append(x)
        for x in random_choice[0][sensitive_index:]:
            test_b1.append(x)
        # Case2
        random2_list = random.choices(choice_list, k=2)
        test_a2_ = random2_list[0][sensitive_index:]
        test_b2_ = random2_list[1][sensitive_index:]

        test_a2 = [i for i in random2_list[0][:sensitive_index]]
        test_b2 = [i for i in random2_list[1][:sensitive_index]]

        for x in test_b2_:
            test_a2.append(x)
        for x in test_a2_:
            test_b2.append(x)
        new_test_sub = np.array([test_a1, test_b1, test_a2, test_b2])

        list_index_new = []
        list_index_new2 = []
        for i in range(len(new_test_sub)):
            x2 = []
            #print(new_test_sub[i])
            for index in range(len(new_test_sub[i])):
                val = new_test_sub[i][index]
                #print(new_test_sub[i][index])
                if new_test_sub[i][index] == 1 and index == sensitive_index:
                    val = 0
                elif new_test_sub[i][index] == 0 and index == sensitive_index:
                    val = 1
                x2.append(val)
            x2 = np.array(x2)
            # print(new_x_test_sub[i], x2)
            if system_type == 'A':
                pred_x1 = np.sign(np.dot(model, new_test_sub[i][:20]))
                pred_x2 = np.sign(np.dot(model, x2[:20]))
            else:
                pred_x1 = model.predict(np.reshape(new_test_sub[i][:20], (-1, 20)))
                pred_x2 = model.predict(np.reshape(x2[:20], (-1, 20)))
            #if isinstance(pred_x1, )
            if pred_x1 != pred_x2: # change back to  pred_x1[0] != pred_x2[0]:
                # p_ += 1
                list_index_new.append(i)
            else:
                list_index_new2.append(i)
        new_str = [str(v) for v in new_test_sub[i]]
        new_str = ','.join(new_str)
        if not new_str in list_children:
            list_children.append(new_str)
            list_test_suites.append(new_test_sub[i])
        print('Children generated: ', len(list_children))






