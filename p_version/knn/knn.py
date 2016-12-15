import numpy as np

class KdNode(object):
    """
    the kdNode of a Kd-tree
    """
    def __init__(self, elem, split, left, right, parent):
        """
        :param elem: the element data in this node
        :param split: using which split to do split
        :param left:  the left child
        :param right: the right child
        :param parent: the parent
        """
        self.elem = elem
        self.split = split
        self.left = left
        self.right = right
        self.parent = parent


class KdTree(object):
    """
    the KdTree structure, can be used to find the k-nearest elements
    """
    def __init__(self, data):
        """
        construct a kd-tree
        :param data: the data used construct a kd-tree
        """
        def buildtree(data, depth, parent):
            """
            the building tree function
            :param data: data used to do the building job. data is a 2d array numpy object. which is (m_samples, n_features)
            :param depth: used to decide the split. do it around
            :param parent: it's parent
            :return:
            """
            [m_samples, n_features] = np.shape(data)
            # decide the split
            split = depth % n_features
            # sort the data according this split
            idx = np.argsort(data[:, split], kind='mergesort')
            data[:, :] = data[idx, :]
            # get the median as the elem
            elem = data[m_samples/2, :]

            root = KdNode(elem=elem, split=split, left=None, right=None, parent=parent)
            root.left = buildtree(data[:m_samples/2, :], depth+1, parent=root)
            root.right = buildtree(data[m_samples/2 + 1:, :], depth+1, parent=root)

            return root

        # convert the data to 2d array
        data = np.array(data, dtype=np.float32)
        [m_samples, n_features] = np.shape(data)
        self.tree = buildtree(data, 0, None)
        self.m_samples = m_samples
        self.n_features = n_features


    def search_knn(self, target, k):
        pass