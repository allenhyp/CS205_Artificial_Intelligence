from sets import Set


class Queue(object):
    def __init__(self):
        self.items = []

    def is_empty(self):
        return self.items == []

    def enqueue(self, item):
        for i in item:
            self.items.insert(0, i)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)


class General_Search(object):
    def __init__(self):
        self.moves = {0: [1, 3], 1: [2, 4, 0], 2: [1, 5], 3: [0, 4, 6], 4: [5, 7, 1, 3],
                      5: [8, 4, 2], 6: [3, 7], 7: [8, 6, 4], 8: [5, 7]}
        self.goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        self.goal_state_string = '123456780'
        self.duplicated_state = Set()

    def make_queue(self, state):
        self.nodes = Queue()
        self.nodes.enqueue([[self.state_to_string(state), 0]])
        self.max_size_of_queue = 1
        self.num_of_expanded_nodes = 0
        return self.nodes

    def expand(self, node):
        print(node)
        self.display(node[0])
        print("Expanding this node...")
        zero_cur_pos = node[0].find('0')
        zero_next_pos = self.moves[zero_cur_pos]
        new_nodes = []
        for z in zero_next_pos:
            temp_node = node[0][:z] + '0' + node[0][z + 1:]
            temp_node = temp_node[:zero_cur_pos] + node[0][z] + temp_node[zero_cur_pos + 1:]
            if temp_node not in self.duplicated_state:
                new_nodes.append([temp_node, node[1] + 1])
                self.duplicated_state.add(temp_node)
            self.num_of_expanded_nodes += len(new_nodes)
        return new_nodes

    def uniform_cost_search(self, original_state_string):
        self.make_queue(original_state_string)

        while True:
            if self.nodes.is_empty():
                print('FAILURE\n')
                break
            node = self.nodes.dequeue()
            if node[0] == self.goal_state_string:
                print('SUCCESS\n')
                break

            self.nodes.enqueue(self.expand(node))
            self.update_queue_size()

        print("Expanded nodes = {},\nmax num in queue = {},\ndepth = {}\n"
              .format(self.num_of_expanded_nodes,
                      self.max_size_of_queue,
                      node[1]))
        return

    def a_star_misplaced_tile_heuristic(self, original_state_string):

        return

    def a_star_manhattan_distance_heuristic(self, original_state_string):
        return True

    def state_to_string(self, state):
        state_str = ''
        for s in state:
            state_str += s
        return state_str

    def update_queue_size(self):
        self.max_size_of_queue = max(self.max_size_of_queue, self.nodes.size())

    def display(self, s):
        print('{} {} {}'.format(s[0], s[1], s[2]))
        print('{} {} {}'.format(s[3], s[4], s[5]))
        print('{} {} {}'.format(s[6], s[7], s[8]))


def main():
    '''
    print("Enter your puzzle, use a zero to represent the blank\n")
    original_state = raw_input("Enter the first row, use space between numbers  \t")
    original_state += raw_input("Enter the second row, use space between numbers \t")
    original_state += raw_input("Enter the third row, use space between numbers  \t")
    print("Enter your choice of algorithm\n")
    print("\t1. Uniform Cost\n")
    print("\t2. A* with the Misplaced Tile heuristic\n")
    print("\t3. A* with the Manhattan distance heuristic\n")
    '''
    original_state = '1 2 3 4 5 6 8 7 0'
    choice = raw_input()
    original_state_string = ''
    for s in original_state.split(' '):
        original_state_string += s
    solve_eight_puzzle = General_Search()
    if choice == '1':
        solve_eight_puzzle.uniform_cost_search(original_state_string)
    elif choice == '2':
        solve_eight_puzzle.a_star_misplaced_tile_heuristic(original_state_string)
    elif choice == '3':
        solve_eight_puzzle.a_star_manhattan_distance_heuristic(original_state_string)
    return


if __name__ == "__main__":
    main()
