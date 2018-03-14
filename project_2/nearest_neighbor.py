from random import randrange
from math import sqrt


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

    def leave_one_out_cross_validation(self, feature_set, upper_bound):
        predict_correct = 0
        predict_wrong = 0
        for test_index in range(self.number_of_test_case):
            # test_index = randrange(self.number_of_instances)
            test_data = self.data_list[test_index]
            min_dis = 1e308
            predict_class = 2
            for train_data in self.data_list:
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
        best_set = ''
        wrong_upper_bound = self.number_of_test_case
        for i in range(1, self.number_of_features + 1):
            print("On the {0}th level of the search tree".format(i))
            feature_to_add_at_this_level = 0
            best_accuracy_this_level = 0.
            for j in range(1, self.number_of_features + 1):
                test_feature_set = set()
                test_feature_set = test_feature_set | self.current_feature_set
                if j not in test_feature_set:
                    test_feature_set.add(j)
                    ret = self.leave_one_out_cross_validation(test_feature_set, wrong_upper_bound)
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
                best_set = ''
                for s in self.current_feature_set:
                    best_set += '{0}, '.format(s)
        print("Overall, the best set with accuracy {0} is [{1}]".format(best_so_far_accuracy, best_set[:-2]))

    def backward_elimination(self):
        best_so_far_accuracy = 0.
        best_set = ''
        wrong_upper_bound = self.number_of_test_case
        self.current_feature_set = set()
        for i in range(1, self.number_of_features + 1):
            self.current_feature_set.add(i)
        best_so_far_accuracy = self.leave_one_out_cross_validation(self.current_feature_set, self.number_of_test_case) / self.number_of_test_case

        for i in range(1, self.number_of_features + 1):
            print("On the {0}th level of the search tree".format(i))
            feature_to_subtract_at_this_level = 0
            best_accuracy_this_level = 0.
            test_feature_set = set()
            for feature in self.current_feature_set:
                test_feature_set.union(self.current_feature_set)
                test_feature_set.difference(feature)
                ret = self.leave_one_out_cross_validation(test_feature_set, wrong_upper_bound)
                new_acc = float(ret) / self.number_of_test_case
                print("--Consider subtracting the {0}th feature => acc = {1}".format(feature, new_acc))
                if new_acc > best_accuracy_this_level:
                    feature_to_subtract_at_this_level = feature
                    best_accuracy_this_level = new_acc
                    wrong_upper_bound = self.number_of_test_case - ret
            print("best_accuracy_this_level: {0}, best_so_far_accuracy: {1}".format(best_accuracy_this_level, best_so_far_accuracy))
            self.current_feature_set.remove(feature_to_subtract_at_this_level)
            print("On level {0}, I subtracted feature {1} to current set".format(i, feature_to_subtract_at_this_level))
            if best_accuracy_this_level >= best_so_far_accuracy:
                best_so_far_accuracy = best_accuracy_this_level
                best_set = ''
                for s in self.current_feature_set:
                    best_set += '{0}, '.format(s)
        print("Overall, the best set with accuracy {0} is [{1}]".format(best_so_far_accuracy, best_set[:-2]))

    def customed_algorithm(self):
        return 0


def main():
    # input_file_name = input("Type in the name of the file to test: ")
    print("Type the number of the algorithm you want to run.")
    print("\t 1) Forward Selection")
    print("\t 2) Backward Elimination")
    print("\t 3) customed Algorithm")
    method_option = input()
    input_file_name = "CS205_BIGtestdata__36.txt"
    nearest_neighbor = Nearest_Neighbor(input_file_name)
    if method_option == '1':
        nearest_neighbor.forward_selection()
    elif method_option == '2':
        nearest_neighbor.backward_elimination()
    elif method_option == '3':
        nearest_neighbor.customed_algorithm()


if __name__ == "__main__":
    main()
