import numpy as np
from math import log2


class Tree:
    def __init__(self, root_node):
        self.root = root_node


    def build_tree(self, node):

        ONE_OUTPUT = 1

        for child in node.children:
            unique_outputs = set(child.data_set[:,-1])
            if ONE_OUTPUT == len(unique_outputs):
                output = unique_outputs.pop()
                child.output = output
                child.output_value_set = True
            else:
                new_labels = np.asarray(child.data_set[:,-1])
                child.set_best_feature(new_labels.reshape((len(new_labels),1)))
                child.create_children_data_sets()
                self.build_tree(child)

    # Print the Tree
    def print_tree(self, node):
        for child_index in range(len(node.children)):
            child = node.children[child_index]
            if  len(child.children) == 0:
                print(f"Child {child_index} Info:\n Output: {child.output}\n Output Value Set: {child.output_value_set}\n")
            else:
                self.print_tree(child)



class Node:
    def __init__(self, data_set, counts):
        self.best_feature = 0
        self.children = []
        self.data_set = data_set
        self.class_col = np.shape(self.data_set)[1] - 1
        self.ds_total_obs = np.shape(self.data_set)[0]
        self.counts = counts
        self.output = None
        self.output_value_set = False


    def __str__(self):
        return str(f'Node:\nBest Feature -> {self.best_feature}\nChildren -> {self.children}\nData Set -> \n{self.data_set}\n')

    def set_best_feature(self, y):
        entropies = []
        test1 = np.shape(self.data_set)[1]
        test2 = np.shape(y)[1]
        for feature in range( test1 - test2 ):

            feature_info = {}

            for observation in range(self.ds_total_obs):
                observation_val = self.data_set[observation][feature]
                class_val = self.data_set[observation][self.class_col]
                if observation_val in feature_info:
                    feature_info[observation_val]['total'] = feature_info[observation_val]['total'] + 1
                    if class_val in feature_info[observation_val]:
                        feature_info[observation_val][class_val] = feature_info[observation_val][class_val] + 1.0
                    else:
                        feature_info[observation_val][class_val] = 1.0
                else:
                    feature_info[observation_val] = {class_val: 1.0, 'total': 1}

            entropy_sum = 0
            for category in feature_info:
                entropy_sum = entropy_sum + self._calc_entropy(feature_info[category])

            entropies.append(entropy_sum)

        min_entropy = min(entropies)
        self.best_feature = entropies.index(min_entropy)

    def create_children_data_sets(self):
        COLUMN = 1
        children_data_sets = {}
        for output_type in range(self.counts[self.best_feature]):

            children_data_sets[output_type] = self.data_set[self.data_set[:, self.best_feature] == output_type]
            child_data = np.delete(children_data_sets[output_type], self.best_feature, COLUMN)
            child_counts = np.delete(self.counts,self.best_feature)

            # Create Child Node
            child = Node(child_data, child_counts)

            # Append Child to Children List
            self.children += [child]

        pass


    def _calc_entropy(self, category_info):
        entropy_sum = 0
        cat_total_obs = category_info['total']
        del category_info['total']

        for type in category_info:
            type_obs = category_info[type]
            prob = type_obs / cat_total_obs
            entropy_sum = entropy_sum + (cat_total_obs / self.ds_total_obs) * (-(prob) * (log2(prob)))

        return entropy_sum

    def check_children_outputs(self, row):

        value_to_check = int(row[self.best_feature])
        child_node_to_check = self.children[value_to_check]

        if child_node_to_check.output_value_set:
            final_output = child_node_to_check.output
        else:
            modified_row = np.delete(row, self.best_feature)
            final_output = child_node_to_check.check_children_outputs(modified_row)

        return final_output
