class Tree:
    def __init__(self, root_node):
        self.root = root_node


    def __str__(self):
        return str(self.root)


class Node:
    def __init__(self, value, data_set):
        self.value = value
        self.children = []
        self.data_set = data_set

    def add_child(self, obj):
        self.children.append(obj)

    def __str__(self):
        return str(f'Node Data: {self.value}')
