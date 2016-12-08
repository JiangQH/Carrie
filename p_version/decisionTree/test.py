from tree import Tree
from helper import SimpleHelper as SH

fid = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fid.readlines()]
lenses_labels = ['age', 'prescript', 'astigmatic', 'tearRate']
tree = Tree().createTree(lenses, lenses_labels)
print tree