from random import randrange


class Nearest_Neighbor(object):
    def __init__(self, file_name):
        self.data_list = []
        self.number_of_instances = 0
        self.number_of_features = 0
        self.current_feature_set = set()
        self.parse_data(file_name)
        self.normalize_data()

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
        for i in range(1, self.number_of_features + 1):
            total = 0
            for d in self.data_list:
                total += d[i]
            for d in self.data_list:
                d[i] = d[i] / total

    def leave_one_out_cross_validation(self, feature_set):
        test_index = randrange(self.number_of_instances)
        test_data = self.data_list[test_index]
        min_dis = 1e308
        predict_correct = 0.
        for __ in range(self.number_of_instances):
            predict_class = 2
            for train_data in self.data_list:
                if test_data != train_data:
                    this_dis = 0
                    for feature in feature_set:
                        this_dis += (train_data[feature] - test_data[feature]) ** 2
                    if this_dis < min_dis:
                        predict_class = train_data[0]
                        min_dis = this_dis
            predict_correct = predict_correct + 1 if predict_class == test_data[0] else predict_correct
        return predict_correct / self.number_of_instances


    def forward_selection(self):
        for i in range(self.number_of_features):
            print("On the {0}th level of the search tree".format(i))
            best_so_far_accuracy = 0.
            feature_to_add_at_this_level = 0
            for j in range(self.number_of_features):
                test_feature_set = set()
                test_feature_set = test_feature_set | self.current_feature_set
                if j not in test_feature_set:
                    test_feature_set.add(j)
                    new_acc = self.leave_one_out_cross_validation(test_feature_set)
                    print("--Consider adding the {0} feature => acc = {1}".format(j, new_acc))
                    if new_acc > best_so_far_accuracy:
                        feature_to_add_at_this_level = j
            self.current_feature_set.add(feature_to_add_at_this_level)
            print("On level {0}, I added feature {1} to current set".format(i, feature_to_add_at_this_level))
            print(self.current_feature_set)

    def backward_selection(self):
        return 0

    def customed_algorithm(self):
        return 0


def main():
    # input_file_name = input("Type in the name of the file to test: ")
    input_file_name = "CS205_BIGtestdata__1.txt"
    nearest_neighbor = Nearest_Neighbor(input_file_name)
    print("Type the number of the algorithm you want to run.")
    print("\t 1) Forward Selection")
    print("\t 2) Backward Selection")
    print("\t 3) customed Algorithm")
    method_option = input()
    if method_option == '1':
        nearest_neighbor.forward_selection()
    elif method_option == '2':
        nearest_neighbor.backward_selection()
    elif method_option == '3':
        nearest_neighbor.customed_algorithm()


if __name__ == "__main__":
    main()
