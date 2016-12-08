import matplotlib.pyplot as plt

class SimpleHelper():

    def createDataSet(self):
        dataSet = [[1, 1, 'yes'],
                   [1, 1, 'yes'],
                   [1, 0, 'no'],
                   [0, 1, 'no'],
                   [0, 1, 'no']]
        labels = ['no surfacing', 'flippers']
    # change to discrete values
        return dataSet, labels

    def plotTree(self):
        # define the box and arrow style
        decision_node = dict(boxstyle="sawtooth", fc="0.8")
        leaf_node = dict(boxstyle="round64", fc="0.8")
        arrow_args = dict(arrowstyle="<-")

        #define the

    def getNumLeafs(self, Tree):
        num_leafs = 0
        first_str = Tree.keys()[0]
        second_dict = Tree[first_str]
        for key in second_dict.keys():
            if type(second_dict[key]).__name__ == 'dict':
                num_leafs += self.getNumLeafs(second_dict[key])
            else:
                num_leafs += 1
        return num_leafs

    def getTreeDepth(self, Tree):
        max_depth = 0
        first_str = Tree.keys()[0]
        second_dict = Tree[first_str]
        for key in second_dict.keys():
            if type(second_dict[key]).__name__ == 'dict':
                this_depth = 1 + self.getTreeDepth(second_dict[key])
            else:
                this_depth = 1
            if this_depth > max_depth:
                max_depth = this_depth
        return max_depth