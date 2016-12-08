"""
This file implements the ID3 Decision Tree / C4.5 /CART
"""
import numpy as np
class Tree:
    def __init__(self):
        pass

    def calEnt(self, dataSet):
        '''
        calculate the Entropy of a given dataset
        :param dataSet:  the dataset to be computed. note that the last column of the dataset has the class label
        :return: the computed entropy
        '''
        total_num = len(dataSet)
        label_counts = {}
        # calculate the label count of each
        for entry in dataSet:
            label = entry[-1]
            if label not in label_counts.keys():
                label_counts[label] = 0
            label_counts[label] += 1

        # compute the entropy
        ent = 0.0
        for key in label_counts:
            prob = label_counts[key].astype(np.float32) / total_num
            ent -= prob * np.log2(prob)

        return ent


    def calGini(self, dataSet):
        """
        calculate the Gini factor
        :param dataSet: the dataset to be computed. note that the last column of the dataset has the class label
        :return: the computed gini factor
        """
        total_num = len(dataSet)
        label_counts = {}
        for entry in dataSet:
            label = entry[-1]
            if label not in label_counts.keys():
                label_counts[label] = 0
            label_counts[label] += 1

        # compute the Gini factor
        temp = 0.0
        for key in label_counts:
            prob = label_counts[key].astype(np.float32) / total_num
            temp += prob * prob

        return 1 - temp




