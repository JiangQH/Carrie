"""
This file implements the ID3 Decision Tree / C4.5 /CART
"""
import numpy as np
import operator
class Tree:

    def __init__(self):
        pass

    def calEnt(self, data_set, axis=-1):
        '''
        calculate the Entropy of a given dataset
        :param dataSet:  the dataset to be computed. note that the last column of the dataset has the class label
        :param axis: with repecte to which to compute the entropy. default it label
        :return: the computed entropy
        '''
        total_num = len(data_set)
        label_counts = {}
        # calculate the label count of each
        for entry in data_set:
            label = entry[axis]
            if label not in label_counts.keys():
                label_counts[label] = 0
            label_counts[label] += 1

        # compute the entropy
        ent = 0.0
        for key in label_counts:
            prob = float(label_counts[key]) / total_num
            ent -= prob * np.log2(prob)

        return ent


    def calGini(self, data_set):
        """
        calculate the Gini factor
        :param dataSet: the dataset to be computed. note that the last column of the dataset has the class label
        :return: the computed gini factor
        """
        total_num = len(data_set)
        label_counts = {}
        for entry in data_set:
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

    def splitData(self, data_set, axis, value):
        """
        get the dataSet with axis value == value
        :param dataSet: the dataset to be split
        :param axis: axis specify which to indicate
        :param value: the desired value
        :return: splited data
        """
        splited = []
        for entry in data_set:
            if entry[axis] == value:
                temp = entry[:axis]
                temp.extend(entry[axis+1:])
                splited.append(temp)
        return splited


    def findBestSplit(self, data_set):
        """
        find the best feature, with most information gain(ID3)
        :param dataSet: the dataset to compute
        :return: axis/index of the best find
        """
        feature_len = len(data_set[0]) - 1
        data_len = len(data_set)
        max_info_gain = -1
        chosen = -1
        baseEnt = self.calEnt(data_set)
        # for each feature
        for axis in range(feature_len):
            feature_holds = [entry[axis] for entry in data_set]
            unique_features = list(set(feature_holds))

            # compute the conditional entropy
            temp = 0.0
            for feature in unique_features:
                sub_dataset = self.splitData(data_set, axis, feature)
                ent = self.calEnt(sub_dataset)
                prob = len(sub_dataset) / float(data_len)
                temp += prob * ent
            info_gain = baseEnt - temp
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                chosen = axis

        return chosen

    def findMostLabel(self, labellist):
        """
        find the most occured label in labellist
        :param labellist:
        :return:
        """
        label_count = {}
        for label in labellist:
            if label not in label_count.keys():
                label_count[label] = 0
            label_count[label] += 1

        sorted_label = sorted(label_count, key=operator.itemgetter(1), reverse=True)
        return sorted_label[0][0]


    def createTree(self, data_set, feature_labels):
        """
        create the Tree. without cutoff. ID3
        :param data_set:
        :param labels:
        :return:
        """
        # get the label of each data point
        labellist = [entry[-1] for entry in data_set]
        if labellist.count(labellist[0]) == len(labellist):
            # if all the label is same. return the label
            return labellist[0]
        if len(data_set[0]) == 1:
            # if all the feature has been used
            return self.findMostLabel(labellist)
        # now find the best feature to split
        best_feature_axis = self.findBestSplit(data_set)
        best_feature_label = feature_labels[best_feature_axis]
        tree = {best_feature_label: {}}
        # split the dataset with this feature
        feature_values = [entry[best_feature_axis] for entry in data_set]
        unique_feature_values = set(feature_values)
        for feature_value in unique_feature_values:
            sub_dataset = self.splitData(data_set, best_feature_axis, feature_value)
            # if sub_dataset is null. return the parent's majority
            if len(sub_dataset) == 0:
                tree[best_feature_label][feature_value] = self.findMostLabel(labellist)
            # recursively create the tree
            else:
                sub_feature_labels = feature_labels[:best_feature_axis]
                sub_feature_labels.extend(feature_labels[best_feature_axis+1:])
                tree[best_feature_label][feature_value] = self.createTree(sub_dataset, sub_feature_labels)

        return tree


    def classify(self, input_tree, feat_labels, test_vec):
        first_str = input_tree.keys()[0]
        second_dict = input_tree[first_str]
        feat_index = feat_labels.index(first_str)
        for key in second_dict.keys():
            if test_vec[feat_index] == key:
                if type(second_dict[key]).__name__ == 'dict':
                    class_label = self.classify(second_dict[key], feat_labels, test_vec)
                else:
                    class_label = second_dict[key]
        return class_label

    def storeTree(self, input_tree, filename):
        import pickle
        with open(filename, 'w') as fid:
            pickle.dump(input_tree, fid)
            fid.close()

    def loadTree(self, filename):
        import pickle
        with open(filename, 'r') as fid:
            tree = pickle.load(fid)
            fid.close()
            return tree









