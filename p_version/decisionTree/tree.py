"""
This file implements the ID3 Decision Tree / C4.5 /CART
"""
import numpy as np
class Tree:
    def __init__(self):
        pass

    def calEnt(self, dataSet, axis=-1):
        '''
        calculate the Entropy of a given dataset
        :param dataSet:  the dataset to be computed. note that the last column of the dataset has the class label
        :param axis: with repecte to which to compute the entropy. default it label
        :return: the computed entropy
        '''
        total_num = len(dataSet)
        label_counts = {}
        # calculate the label count of each
        for entry in dataSet:
            label = entry[axis]
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

    def splitData(self, dataSet, axis, value):
        """
        get the dataSet with axis value == value
        :param dataSet: the dataset to be split
        :param axis: axis specify which to indicate
        :param value: the desired value
        :return: splited data
        """
        splited = []
        for entry in dataSet:
            if entry[axis] == value:
                temp = entry[:axis]
                temp.extend(entry[axis+1:])
                splited.append(temp)
        return splited


    def findBestSplit(self, dataSet):
        """
        find the best feature, with most information gain(ID3)
        :param dataSet: the dataset to compute
        :return: axis/index of the best find
        """
        feature_len = len(dataSet[0]) - 1
        data_len = len(dataSet)
        max_info_gain = -1
        chosen = -1
        baseEnt = self.calEnt(dataSet)
        # for each feature
        for axis in range(feature_len):
            feature_holds = [entry[axis] for entry in dataSet]
            unique_features = list(set(feature_holds))

            # compute the conditional entropy
            temp = 0.0
            for feature in unique_features:
                sub_dataset = self.splitData(dataSet, axis, feature)
                ent = self.calEnt(sub_dataset)
                prob = len(sub_dataset) / float(data_len)
                temp += prob * ent
            info_gain = baseEnt - temp
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                chosen = axis

        return chosen








