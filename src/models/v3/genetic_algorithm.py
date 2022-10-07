import random
from scipy.stats import gaussian_kde, entropy, ks_2samp, norm
from numpy import linspace
import numpy as np
import operator
from tensorflow.python.keras.utils.np_utils import to_categorical

from src.models.v3.NNClassifier import NNClassifier
from src.models.v3.load_data import LoadData
from src.models.v3.sensitivity_utils import check_feature_value_belong_to, is_categorical
from src.models.v3.utility_functions import split_features_target, data_split
import numpy.random as npr

class GeneticAlgorithsm:
    def __init__(self, model, _features_posible_value, target_index, sensitive_indices, sensitive_index_chosen=0, max_test=100000):
        #self.data = data
        self.model = model
        self.path = '../dataset/suites-v2/'
        self.target_index = target_index
        self.test_suites = []
        self.list_Discrimination = []
        self.list_Non_discriminative = []
        self.list_Non_discriminative_str = []
        self.list_alread_checked = []
        self.max_test = max_test
        self.Iteration = 1
        self.pos = 0
        self.neg = 0
        self.sensitive_indices = sensitive_indices
        self.sensitive_index_chosen = sensitive_index_chosen
        self._features_posible_value = _features_posible_value
        self.test_suite_file = open(self.path+'TestSuit_3.txt', 'w+')
    def _inference(self, data):
        return self.model.predict(data)

    def weighted_random_choice(self, population_dict):
        max_val = sum(list(population_dict.keys()))
        pick = random.uniform(0, max_val)
        current = 0
        parent_list = []
        sorted_d = dict(sorted(population_dict.items(), key=operator.itemgetter(1), reverse=True))
        for key, value in sorted_d.items():
            current += key
            parent_list.append(value)
            if current > pick:
                break
        return parent_list
    def check_stoping(self):
        continue_search = True
        if len(self.list_Discrimination) >= self.max_test:
            continue_search = False
        #for test_ in
        return continue_search
    def selectOne(self, population):
        max = sum([c.fitness for c in population])
        selection_probs = [c.fitness / max for c in population]
        return population[npr.choice(len(population), p=selection_probs)]
    def _search(self, original_test_data, new_test_data, y_test_original, y_pred_original, x_test_permutated, y_pred_permutated, value_range=[0.4, 0.6]):
        test_suilts = []
        potential_parent_maximized_error = {}
        potential_parents_value_range = []
        val = 0
        p_sample_pos, p_sample_neg = [], []
        q_sample_pos, q_sample_neg = [], []
        for i in range(len(y_pred_original)):
            p_sample_pos.append(y_pred_original[i][0])
            p_sample_neg.append(y_pred_original[i][1])

            q_sample_pos.append(y_pred_permutated[i][0])
            q_sample_neg.append(y_pred_permutated[i][1])
            arg_max_orig = y_pred_original[i].argmax(axis=-1)
            arg_max_perm = y_pred_permutated[i].argmax(axis=-1)

            if arg_max_orig != arg_max_perm:
                test_suilts.append(new_test_data[i])
            diff = (abs(y_pred_permutated[i][0]-y_pred_original[i][0]) + abs(y_pred_permutated[i][1]-y_pred_original[i][1]))/2

            if diff > val:
                #potential_parent_maximized_error = {}
                potential_parent_maximized_error[diff] = list(new_test_data[i])
                val = diff
            if (y_pred_original[i][0] >= value_range[0] and y_pred_original[i][0] <= value_range[1]) or (y_pred_original[i][1] >= value_range[0] and y_pred_original[i][1] <= value_range[1]):
                potential_parents_value_range.append(original_test_data[i])
            if (y_pred_permutated[i][0] >= value_range[0] and y_pred_permutated[i][0] <= value_range[1]) or (y_pred_permutated[i][1] >= value_range[0] and y_pred_permutated[i][1] <= value_range[1]):
                potential_parents_value_range.append(new_test_data[i])
        p_sample_pos, p_sample_neg = np.array(p_sample_pos), np.array(p_sample_neg)
        q_sample_pos, q_sample_neg = np.array(q_sample_pos), np.array(q_sample_neg)
        min_kl_pos, _, _ = self.search_kl_divergence(p_sample_pos, q_sample_pos)
        min_kl_neg, _, _ = self.search_kl_divergence(p_sample_neg, q_sample_neg)


        return test_suilts, potential_parent_maximized_error, potential_parents_value_range, (min_kl_neg+min_kl_pos)/2
    def _crossover(self, data_dict):
        data_new = []
        for key, val in data_dict.items():
            data_new.extend(val)
        print(' Now performing mutation on dataset: ', len(data_new))
        if len(data_new) > 1:
            data_split = np.array_split(data_new, 2)
            data_1 = data_split[0]
            data_2 = data_split[1]

            ## Todo: Mutation starts here
            ## Todo: 3 multiple crossover
            k = 3
            #if len(data_1[0]) >=10 and len(data_1[0]) <15:
            #    k = 3
            #if len(data_1[0]) > 15:
            #    k = 4
            print(' --- crossover: ', len(data_1), len(data_2))
            list_all_combination = []
            for index in range(len(data_1)):
                if index%500 == 0:
                    print(' --- index: ', index, len(list_all_combination))
                range_list = list(range(1, len(data_1[0])))
                if self.target_index in range_list:
                    range_list = range_list.remove(self.target_index)
                random_points = sorted(random.sample(range_list, k))
                point_0 = random_points[0]
                point_1 = random_points[1]
                point_2 = random_points[2]

                part0_x1 = data_1[index][0:point_0]
                part1_x1 = data_1[index][point_0:point_1]
                part2_x1 = data_1[index][point_1:point_2]
                part3_x1 = data_1[index][point_2:]
                #print(data_1[index], 'range points', random_points, part0_x1, part1_x1, part2_x1, part3_x1)
                for index2 in range(len(data_2)):
                    #part 0
                    part0_x2 = data_2[index2][0:point_0]
                    if np.array_equal(part0_x1, part0_x2) == False:
                        new_x1 = np.concatenate([part0_x2, data_1[index][point_0:]])
                        new_x2 = np.concatenate([part0_x1, data_2[index2][point_0:]])
                        #list_all_combination.append(new_x1)
                        #list_all_combination.append(new_x2)
                        #print('first: ', len(new_x1), len(new_x2), new_x1, part0_x2, data_1[index][point_0:])
                        list_all_combination.append(list(new_x1))
                        list_all_combination.append(list(new_x2))

                    # part 1
                    part1_x2 = data_2[index2][point_0:point_1]

                    if np.array_equal(part1_x1, part1_x2) == False:
                        new_x1 = np.concatenate([np.concatenate([part0_x1, part1_x2]), data_1[index][point_1:]])
                        new_x2 = np.concatenate([np.concatenate([part0_x2, part1_x1]), data_2[index2][point_1:]])
                        #print('second: ', len(new_x1), len(new_x2), new_x1, )
                        #list_all_combination.append(new_x1)
                        #list_all_combination.append(new_x2)
                        list_all_combination.append(list(new_x1))
                        list_all_combination.append(list(new_x2))

                    # part 2
                    part2_x2 = data_2[index2][point_1:point_2]
                    if np.array_equal(part2_x1, part2_x2) == False:
                        new_x1 = np.concatenate([np.concatenate([data_1[index][0:point_1],part2_x2]), data_1[index][point_2:]])
                        new_x2 = np.concatenate([np.concatenate([data_2[index2][0:point_1],part2_x1]), data_2[index2][point_2 :]])
                        #list_all_combination.append(new_x1)
                        #list_all_combination.append(new_x2)
                        list_all_combination.append(list(new_x1))
                        list_all_combination.append(list(new_x2))
                        #print('third: ', len(new_x1), len(new_x2), new_x1)

                    # part 3
                    part3_x2 = data_2[index2][point_2:]
                    if np.array_equal(part3_x1, part3_x2) == False:
                        new_x1 = np.concatenate([data_1[index][0:point_2],part3_x2])
                        new_x2 = np.concatenate([data_2[index2][0:point_2],part3_x1])
                        list_all_combination.append(list(new_x1))
                        list_all_combination.append(list(new_x2))
                        #print('fourth: ', len(new_x1), len(new_x2), new_x1)

                    new_x_combine_1 = np.concatenate([np.concatenate([part0_x1, part1_x2]), np.concatenate([part2_x1, part3_x2])])
                    new_x_combine_2 = np.concatenate([np.concatenate([part0_x2, part1_x1]),
                                                     np.concatenate([part2_x2, part3_x1])])

                    #print('combined: ', len(new_x_combine_1), len(new_x_combine_2), new_x_combine_1)
                    list_all_combination.append(list(new_x_combine_1))
                    list_all_combination.append(list(new_x_combine_2))


                #x_0_0 = np.concatenate((data_1[index][0:, 0:point_0], data_1[index][point_0:, point_0:point_1]), axis=1)
                #print(x.shape)
                #return x, data[0:, index]
            new_data = []
            for list_combination_ in list_all_combination:
                new_str = [str(v) for v in list_combination_]
                new_str = ','.join(new_str)
                if not new_str in self.list_alread_checked:
                    self.list_alread_checked.append(new_str)
                    #potential_parent_maximized_error_new[key] = val
                    new_data.append(list_combination_)
            #new_data = np.array(new_data)
            #new_data = np.reshape(new_data, (-1, 2))
            if len(new_data) > 0 and self.check_stoping():
                print('Iteration: ', self.Iteration, 'New-children: ', len(new_data))
                column_id = self.sensitive_indices[self.sensitive_index_chosen]
                self.Iteration += 1
                self.search_strategy(new_data, strata=4)

    def search_strategy(self, data,strata=4):
        feature_index = self.sensitive_indices[self.sensitive_index_chosen]
        #print('data before split: ', data)
        data_split = np.array_split(data, strata)
        kl_max_data = {}
        selected_parent_dict_all = {}
        kl_max = 0
        index = 0
        for sub_data in data_split:
            #sub_data = sub_data.reshape(-1, len(sub_data[0]))
            #print('sub_data: ', sub_data, sub_data.shape)
            x_test_original, y_test_original = split_features_target(sub_data, self.target_index)
            y_pred_original = self._inference(x_test_original)
            new_test_data = []
            for row_index in range(len(sub_data)):
                new_str = [str(v) for v in sub_data[row_index]]
                new_str = ','.join(new_str)
                self.list_alread_checked.append(new_str)
                row = []
                for j in range(len(sub_data[row_index])):
                    temp_val = sub_data[row_index][j]
                    if j == feature_index:
                        temp_val = check_feature_value_belong_to(self._features_posible_value, sub_data[row_index][j])
                    row.append(temp_val)
                if len(row) > 0:
                    new_test_data.append(row)
            new_test_data = np.array(new_test_data)
            x_test_permutated, _ = split_features_target(new_test_data, self.target_index)
            y_pred_permutated = self._inference(x_test_permutated)
            # print(new_test_data)
            test_suilts, potential_parent_maximized_error, potential_parents_value_range, kl_divergence = self._search(sub_data,new_test_data,y_test_original,y_pred_original,x_test_permutated,y_pred_permutated)
            potential_parents_value_range_new = []
            potential_parent_maximized_error_new = {}
            if kl_divergence > kl_max:
                kl_max = kl_divergence
                kl_max_data[kl_max] = new_test_data
            for key, val in potential_parent_maximized_error.items():
                new_str = [str(v) for v in val]
                new_str = ','.join(new_str)
                if not new_str in self.test_suites:
                    potential_parent_maximized_error_new[key] = val
            for potential_parents in potential_parents_value_range:
                new_str = [str(v) for v in potential_parents]
                new_str = ','.join(new_str)
                if not new_str in self.test_suites:
                    self.test_suites.append(new_str)
                    out = potential_parents[-1]
                    if out > 0:
                        self.pos += 1
                    else:
                        self.neg += 1

                    self.list_Discrimination.append(potential_parents)
                    self.test_suite_file.write(new_str+'\n')
                    self.list_alread_checked.append(new_str)
                    potential_parents_value_range_new.append(potential_parents)
            for test_ in test_suilts:
                new_str = [str(v) for v in test_]
                new_str = ','.join(new_str)
                if not new_str in self.test_suites:
                    self.test_suites.append(new_str)
                    out = potential_parents[-1]
                    if out > 0:
                        self.pos += 1
                    else:
                        self.neg += 1
                    self.list_Discrimination.append(test_)
                    self.test_suite_file.write(new_str + '\n')
                    self.list_alread_checked.append(new_str)

            print('potential_parent_maximized_error: ', len(potential_parent_maximized_error), 'potential_parent_maximized_error_new: ',len(potential_parent_maximized_error_new), 'potential_parents_value_range: ', len(potential_parents_value_range), 'KL Divergence: ', kl_divergence)
            #sorted_d = dict(sorted(potential_parent_maximized_error.items(), key=operator.itemgetter(1), reverse=True))
            selected_parent = self.weighted_random_choice(potential_parent_maximized_error_new)
            #half_data = len(sorted_d)/2
            list_Non_discriminative_str = []
            selected_parent_final = []
            for val in selected_parent:
                # list_potential.append(val)
                # if index <= half_data:
                #    list_potential.append(val)
                new_str = [str(v) for v in val]
                new_str = ','.join(new_str)
                list_Non_discriminative_str.append(new_str)
                selected_parent_final.append(val)
                #self.list_alread_checked.append(new_str)

            #index = 0
            if len(potential_parents_value_range) > 0:
                if len(potential_parents_value_range)==1:
                    range_l = 1
                else:
                    range_l = list(range(1, len(potential_parents_value_range)))
                potential_choices = random.choices(potential_parents_value_range_new, k=random.choice(range_l))
                for potential_ in potential_choices:
                    new_str = [str(v) for v in potential_]
                    new_str = ','.join(new_str)
                    if not new_str in list_Non_discriminative_str:
                        list_Non_discriminative_str.append(new_str)
                        selected_parent_final.append(potential_)
            selected_parent_dict_all[index] = selected_parent_final
            print(' -- sub parents selected: ', len(selected_parent_final), ', from ', len(potential_parents_value_range)+len(potential_parents_value_range))
            index += 1

        self._crossover(selected_parent_dict_all)
        #final_choice_parent = []
        #for key, val in selected_parent_dict_all.items():
        #    if len(val) > 0:
        #        potential_choices = random.choices(val,
        #                                           k=random.uniform(1, len(val)))
        #        for potential_ in potential_choices:
        #            final_choice_parent.append(potential_)
    def search_kl_divergence(self,p_samples, q_samples, MAX_ENTROPY_ALLOWED=1e6):
        n1 = len(p_samples)
        n2 = len(q_samples)
        if n1 == 0 or n2 == 0:
            return 0
        pdf1 = gaussian_kde(p_samples)
        pdf2 = gaussian_kde(q_samples)

        # Calculate the interval to be analyzed further
        a = min(min(p_samples), min(q_samples))
        b = max(max(p_samples), max(q_samples))

        # Plot the PDFs
        lin = linspace(a, b, max(n1, n2))
        p = pdf1.pdf(lin)
        q = pdf2.pdf(lin)
        min_kl=min(MAX_ENTROPY_ALLOWED, entropy(p, q))
        ks_2samp_stats, pValue = ks_2samp(p, q)

        return min_kl, ks_2samp_stats, pValue
    def search_boundery(self):
        pass
    def alter_feature_value_continous(self, test_data, posible_values={}, feature_index=0):
        new_test_data = []
        for i in range(len(test_data)):
            row = []
            for j in range(len(test_data[i])):
                temp_val = test_data[i][j]
                if j == feature_index:
                    temp_val = check_feature_value_belong_to(posible_values, test_data[i][j])
                row.append(temp_val)
            if len(row) > 0:
                new_test_data.append(row)
        # print(new_test_data)
        return np.array(new_test_data, dtype=int)
class TestCaseGeneration:
    def __init__(self, data, sensitive_list, sensitive_indices, target_index):
        self.data = data
        self.sensitive_list = sensitive_list
        self.sensitive_indices = sensitive_indices
        self.target_index = target_index
        self.data_init(data)

    def data_init(self, data):
        self.train, self.test = data_split(data=data, sample_size=0.30)
        self.x_train, self.y_train = split_features_target(self.train, index=self.target_index)
        self.x_test, self.y_test = split_features_target(self.test, index=self.target_index)

        print('self.y_test: ', self.y_test)
    def get_categorical_features_posible_value(self, feature_index):
        print('column data: ', set(self.data[0:, feature_index]))
        data_folded = {}
        for val in list(set(self.data[0:, feature_index])):
            data_folded[val] = [val]
        return data_folded #list(set(self.data[0:, feature_index]))
    def determine_range(self, feature_index, k_folds=3):
        data_range = self.data[0:, feature_index]
        #interval_ = round(((max(sorted_)-min(sorted_))/k_folds),0)
        fold_count = 0
        folded_data = {}
        percentile_25 = np.percentile(data_range, 25)
        percentile_50 = np.percentile(data_range, 50)
        percentile_75 = np.percentile(data_range, 75)
        percentile_100 = np.percentile(data_range, 100)
        if percentile_50 == percentile_25 or percentile_25 == np.min(data_range):
            percentile_25 = np.percentile(np.unique(data_range), 25)
            percentile_50 = np.percentile(np.unique(data_range), 50)
            percentile_75 = np.percentile(np.unique(data_range), 75)
        if percentile_50 == percentile_25:
            percentile_50 = np.max(data_range)/2
        if percentile_25 == np.min(data_range):
            percentile_25 = percentile_50/2
        for i in range(len(data_range)):
            fold_id = percentile_50
            if data_range[i] <= percentile_25:
                fold_id = percentile_25
            elif data_range[i] > percentile_25 and data_range[i] <= percentile_50:
                fold_id = percentile_50
            elif data_range[i] > percentile_50:
                fold_id = percentile_75
            if fold_id in folded_data.keys():
                folded_data[fold_id].add(data_range[i])
            else:
                folded_data[fold_id] = set([data_range[i]])
        for key, val in folded_data.items():
            if len(val) < 3:
                print('Category <3: ', np.min(list(val)), percentile_25, percentile_50, percentile_75, percentile_100)
                if key == percentile_25:
                    val2 = []
                    val2.append(random.randrange(0, np.min(list(val))))
                    val2.append(random.randrange(0, np.min(list(val))))
                    val2.extend(list(val))
                if key == percentile_50:
                    val2 = []
                    val2.append(random.randrange(percentile_50, np.min(list(val))))
                    val2.append(random.randrange(percentile_50, np.min(list(val))))
                    val2.extend(list(val))
                if key == percentile_75:
                    val2 = []
                    val2.append(random.randrange(percentile_25, percentile_100))
                    val2.append(random.randrange(percentile_25, percentile_100))
                    val2.extend(list(val))
                folded_data[key] = list(val)
            else:
                folded_data[key] = list(val)
        return folded_data #list(folded_data.values())

    def model_train(self):
        y_train = to_categorical(self.y_train)
        y_test = to_categorical(self.y_test)
        n_classes = y_test.shape[1]
        NNmodel = NNClassifier(self.x_train.shape[1], n_classes)
        NNmodel.fit(self.x_train, y_train)

        #p_pred = NNmodel.predict(self.x_test)
        column_id = self.sensitive_indices[0]
        algo = GeneticAlgorithsm(NNmodel,self.get_categorical_features_posible_value(column_id),target_index,self.sensitive_indices,0)
        if is_categorical(self.test,column_id):
            algo.search_strategy(self.test, strata=4)
        #else:

        #algo.search_strategy()

if __name__ == '__main__':
    path = '../dataset/'
    ### Adult dataset
    # target_column = 'Probability'
    correlation_threshold = 0.45  # 0.35
    loadData = LoadData(path, threshold=correlation_threshold)  # ,threshold=correlation_threshold
    data_name = 'adult-45'  # _35_threshold

    df_adult = loadData.load_adult_data('adult.data.csv')
    # df_adult = loadData.load_credit_defaulter('default_of_credit_card_clients.csv')
    # df_adult = loadData.load_student_data('Student.csv')
    # df_adult = loadData.load_student_data('Student.csv')
    sensitive_list = loadData.sensitive_list
    sensitive_indices = loadData.sensitive_indices
    colums_list = df_adult.columns.tolist()
    target_name = loadData.target_name
    target_index = loadData.target_index

    testGeneration = TestCaseGeneration(df_adult.to_numpy(),sensitive_list,sensitive_indices,target_index)
    testGeneration.model_train()

