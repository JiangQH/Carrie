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

    def computedist(self, target):
        """
        the distance between the target and this node
        :param target:
        :return:
        """
        return np.linalg.norm(target - self.elem)

    def childs(self):
        result = []
        if self.left:
            result.append(self.left)
        if self.right:
            result.append(self.right)
        return result

    def is_intersects(self, target, dist):
        return np.abs(self.elem[self.split] - target[self.split]) <= dist


class KdTree(object):
    """
    the KdTree structure, can be used to find the k-nearest elements
    """
    def __init__(self, data):
        """
        construct a kd-tree
        :param data: the data used construct a kd-tree
        """
        def _buildtree(data, depth, parent):
            """
            the building tree function
            :param data: data used to do the building job. data is a 2d array numpy object. which is (m_samples, n_features)
            :param depth: used to decide the split. do it around
            :param parent: it's parent
            :return:
            """
            [m_samples, n_features] = np.shape(data)
            if m_samples == 0:
                return None

            # decide the split
            split = depth % n_features
            # sort the data according this split
            idx = np.argsort(data[:, split], kind='mergesort')
            data[:, :] = data[idx, :]
            # get the median as the elem
            elem = data[m_samples/2, :]

            root = KdNode(elem=elem, split=split, left=None, right=None, parent=parent)
            root.left = _buildtree(data[:m_samples/2, :], depth+1, parent=root)
            root.right = _buildtree(data[m_samples/2 + 1:, :], depth+1, parent=root)

            return root

        # convert the data to 2d array
        data = np.array(data, dtype=np.float32)
        [m_samples, n_features] = np.shape(data)
        self.root = _buildtree(data, 0, None)
        self.m_samples = m_samples
        self.n_features = n_features


    def search_knn(self, target, k):
        """
        using kd-tree to search the nearest k members
        :param target: the target searching value
        :param k: k nums
        :return: k nearest neighbours with it's distance
        """
        if k > self.m_samples:
            raise ValueError("k is larger than the total number")
        target = np.array(target, dtype=np.float32)
        if self.n_features != len(target):
            raise ValueError("target feature not agree with the training set")

        current_node = self.root
        prev = None
        while current_node:
            split = current_node.split
            elem = current_node.elem
            prev = current_node
            if target[split] <= elem[split]:
                # go left
                current_node = current_node.left
            else:
                # go right
                current_node = current_node.right

        # now the prev holds the leaf node. we do the back search to hold the k nearest
        if prev is None:
            return []

        # recording the result and nodes we have visited
        result = {}
        visited = set()
        back_current = prev
        while back_current:
            self._check_node(back_current, target, k, result, visited)
            back_current = back_current.parent

        return [(loc.elem.tolist(), value) for loc, value in sorted(result.items(), key=lambda a: a[1])]

    def _check_node(self, current_node, target, k, result, visited):
        visited.add(current_node)
        # compute the distance
        dist = current_node.computedist(target)
        # if the result is not full of k, we add it to the result
        if len(result) < k:
            result[current_node] = dist
        else:
            # the result len is equal to k, we compare the dist with the max
            # if it is less than the max, we add it and remove the max_node
            max_node, max_dist = sorted(result.items(), key=lambda a:a[1], reverse=True)[0]
            if dist < max_dist:
                result[current_node] = dist
                result.pop(max_node)

        # get the new max_dist
        max_node, max_dist = sorted(result.items(), key=lambda a:a[1], reverse=True)[0]
        # for every child of current, we should decide whether to go into it. according
        # to the max_dist now, for it can be better than the max now
        for child in current_node.childs():
            if child in visited:
                continue
            visited.add(child)
            # compute whether the target with the child's subspace
            intersects = child.is_intersects(target, max_dist)
            if intersects:
                # if it is intersects, do the check for it too
                self._check_node(child, target, k, result, visited)



def test():
    data = [[2, 3], [5, 4], [9, 6],
            [4, 7], [8, 1], [7, 2]]
    tree = KdTree(data)
    result = tree.search_knn([2, 4.5], 3)
    print result

if __name__ == "__main__":
    test()
