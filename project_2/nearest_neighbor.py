from random import sample
from math import sqrt
from time import time


class Nearest_Neighbor(object):
    def __init__(self, file_name):
        self.data_list = []
        self.number_of_instances = 0
        self.number_of_features = 0
        self.current_feature_set = set()
        self.parse_data(file_name)
        self.normalize_data()
        self.number_of_test_case = int(self.number_of_instances * 1)
        print("Number of test case = {0}".format(self.number_of_test_case))

    def parse_data(self, file_name):
        file_object = open('./205_proj2_data/' + file_name, 'r')
        data = file_object.readlines()
        for r_data in data:
            new_data = r_data.split('  ')
            data_line = []
            data_line.append(int(new_data[1][1]))
            for i in range(2, len(new_data)):
                new_data_number = float(new_data[i])
                data_line.append(new_data_number)
            self.data_list.append(data_line)
        self.number_of_instances = len(self.data_list)
        self.number_of_features = len(self.data_list[0]) - 1

    def normalize_data(self):
        print("Please wait while I normalize the data...")
        for i in range(1, self.number_of_features + 1):
            total = 0.
            for d in self.data_list:
                total += d[i]
            mean = total / self.number_of_instances
            variance = 0.
            for d in self.data_list:
                variance += (d[i] - mean) ** 2
            sigma = sqrt(variance / (self.number_of_instances - 1))
            for d in self.data_list:
                d[i] = (d[i] - mean) / sigma
        print("Done!")

    def resample(self):
        print("Resamle")
        sample_list = sample(range(self.number_of_instances), k=int(self.number_of_instances*0.95))
        sample_instance = []
        for s in sample_list:
            sample_instance.append(self.data_list[s])
        return sample_instance

    def leave_one_out_cross_validation(self, data_set, feature_set, upper_bound):
        predict_correct = 0
        predict_wrong = 0
        for test_index in range(len(data_set)):
            # test_index = randrange(self.number_of_instances)
            test_data = data_set[test_index]
            min_dis = 1e308
            predict_class = 2
            for train_data in data_set:
                if test_data != train_data:
                    this_dis = 0
                    for feature in feature_set:
                        this_dis += (train_data[feature] - test_data[feature]) ** 2
                    if this_dis < min_dis:
                        predict_class = train_data[0]
                        min_dis = this_dis
            if predict_class == test_data[0]:
                predict_correct += 1
            else:
                predict_wrong += 1
            if predict_wrong >= upper_bound:
                break
        return predict_correct

    def forward_selection(self):
        best_so_far_accuracy = 0.
        best_set = set()
        wrong_upper_bound = self.number_of_test_case
        for i in range(1, self.number_of_features + 1):
            print("On the {0}th level of the search tree".format(i))
            feature_to_add_at_this_level = 0
            best_accuracy_this_level = 0.
            for j in range(1, self.number_of_features + 1):
                test_feature_set = self.current_feature_set.copy()
                if j not in test_feature_set:
                    test_feature_set.add(j)
                    ret = self.leave_one_out_cross_validation(self.data_list, test_feature_set, wrong_upper_bound)
                    new_acc = float(ret) / self.number_of_test_case
                    print("--Consider adding the {0}th feature => acc = {1}".format(j, new_acc))
                    if new_acc > best_accuracy_this_level:
                        feature_to_add_at_this_level = j
                        best_accuracy_this_level = new_acc
                        wrong_upper_bound = self.number_of_test_case - ret
            print("best_accuracy_this_level: {0}, best_so_far_accuracy: {1}".format(best_accuracy_this_level, best_so_far_accuracy))
            self.current_feature_set.add(feature_to_add_at_this_level)
            print("On level {0}, I added feature {1} to current set".format(i, feature_to_add_at_this_level))
            if best_accuracy_this_level > best_so_far_accuracy:
                best_so_far_accuracy = best_accuracy_this_level
                best_set = best_set.union(self.current_feature_set)
            # print("now the best set is {}".format(best_set))
        print("Overall, the best set with accuracy {0} is {1}".format(best_so_far_accuracy, best_set))

    def backward_elimination(self):
        best_so_far_accuracy = 0.
        best_set = set()
        wrong_upper_bound = self.number_of_test_case
        self.current_feature_set = set()
        for i in range(1, self.number_of_features + 1):
            self.current_feature_set.add(i)
            best_set.add(i)
        best_so_far_accuracy = self.leave_one_out_cross_validation(self.data_list, self.current_feature_set, self.number_of_test_case) / self.number_of_test_case
        print("init run with all feature, acc = {0}".format(best_so_far_accuracy))
        for i in range(1, self.number_of_features):
            print("On the {0}th level of the search tree".format(i))
            feature_to_subtract_at_this_level = 0
            best_accuracy_this_level = 0.
            test_feature_set = set()
            for feature in self.current_feature_set:
                test_feature_set = test_feature_set.union(self.current_feature_set)
                test_feature_set.remove(feature)
                ret = self.leave_one_out_cross_validation(self.data_list, test_feature_set, wrong_upper_bound)
                new_acc = float(ret) / self.number_of_test_case
                print("--Consider subtracting the {0}th feature => acc = {1}".format(feature, new_acc))
                if new_acc > best_accuracy_this_level:
                    feature_to_subtract_at_this_level = feature
                    best_accuracy_this_level = new_acc
                    wrong_upper_bound = self.number_of_test_case - ret
            print("best_accuracy_this_level: {0}, best_so_far_accuracy: {1}".format(best_accuracy_this_level, best_so_far_accuracy))
            self.current_feature_set.remove(feature_to_subtract_at_this_level)
            print("On level {0}, I subtracted feature {1} from current set".format(i, feature_to_subtract_at_this_level))
            if best_accuracy_this_level >= best_so_far_accuracy:
                best_so_far_accuracy = best_accuracy_this_level
                best_set = best_set.intersection(self.current_feature_set)
            # print("now the best set is {}".format(best_set))
        print("Overall, the best set with accuracy {0} is {1}".format(best_so_far_accuracy, best_set))

    def search(self, sample, sample_rate, ignore_set=set()):
        current_feature_set = set()
        best_set = set()
        best_so_far_accuracy = 0.
        for i in range(1, sample_rate):
            print("On the {0}th level of the search tree".format(i))
            wrong_upper_bound = len(sample)
            feature_to_add_at_this_level = 0
            best_accuracy_this_level = 0.
            for j in range(1, self.number_of_features + 1):
                test_feature_set = current_feature_set.copy()
                if j not in test_feature_set and j not in ignore_set:
                    test_feature_set.add(j)
                    ret = self.leave_one_out_cross_validation(sample, test_feature_set, wrong_upper_bound)
                    new_acc = float(ret) / self.number_of_test_case
                    print("--Consider adding the {0}th feature => acc = {1}".format(j, new_acc))
                    if new_acc > best_accuracy_this_level:
                        feature_to_add_at_this_level = j
                        best_accuracy_this_level = new_acc
                        wrong_upper_bound = self.number_of_test_case - ret
            print("best_accuracy_this_level: {0}, best_so_far_accuracy: {1}".format(best_accuracy_this_level, best_so_far_accuracy))
            current_feature_set.add(feature_to_add_at_this_level)
            print("On level {0}, I added feature {1} to current set".format(i, feature_to_add_at_this_level))
            if best_accuracy_this_level > best_so_far_accuracy:
                best_so_far_accuracy = best_accuracy_this_level
                best_set = best_set.union(current_feature_set)
        return best_set

    def my_algorithm(self):
        resample_result_list = []
        for _ in range(3):
            sample = self.resample()
            resample_result_list.append(self.search(sample, 10))
        
        strong_feature_set = resample_result_list[0]
        for i in range(1, len(resample_result_list)):
            strong_feature_set = strong_feature_set.intersection(resample_result_list[i])
        resample_result_list = []

        for _ in range(3):
            sample = self.resample()
            resample_result_list.append(self.search(sample, 10, strong_feature_set))

        weak_feature_set = resample_result_list[0].copy()
        for i in range(1, len(resample_result_list)):
            weak_feature_set = weak_feature_set.intersection(resample_result_list[i])

        strong_feature_set = strong_feature_set.union(weak_feature_set)
        ret = self.leave_one_out_cross_validation(self.data_list, strong_feature_set, len(self.data_list))
        acc = float(ret) / len(self.data_list)
        print("Overall, the best set with accuracy {0} is {1}".format(acc, strong_feature_set))


def main():
    # input_file_name = input("Type in the name of the file to test: ")
    print("Type the number of the algorithm you want to run.")
    print("\t 1) Forward Selection")
    print("\t 2) Backward Elimination")
    print("\t 3) My Algorithm")
    method_option = input()
    input_file_name = "CS205_BIGtestdata__41.txt"
    start_time = time()
    nearest_neighbor = Nearest_Neighbor(input_file_name)
    if method_option == '1':
        nearest_neighbor.forward_selection()
    elif method_option == '2':
        nearest_neighbor.backward_elimination()
    elif method_option == '3':
        nearest_neighbor.my_algorithm()
    print("Time = {0}".format(int((time() - start_time) * 1000)))

if __name__ == "__main__":
    main()
