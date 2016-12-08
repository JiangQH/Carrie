from svm import SVM
import numpy as np
from tools import tools
from sklearn.model_selection import  train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import time
# test simple toll, passed

dh = tools.DataHandler()
dataArr, labelArr = dh.loadDataSet('../data/testSet.txt')
classifier = SVM(C=1.0)
classifier.fit(dataArr, labelArr)
b = classifier.getb()
alpha = classifier.getalphas()
#alpha = alpha[alpha > 0]
print b
print np.shape(alpha)
w = 0
for i in range(100):
    w += alpha[i] * labelArr[i] * dataArr[i]
    if alpha[i] > 0.0 and alpha[i] < 0.6:
        print dataArr[i], labelArr[i]
print w


# The test the digit_recognizer
dh = tools.DataHandler()
train_X, train_y, test_X = dh.load_data('../data/train.csv', '../data/test.csv')
# due to here only support binary classification, so we only pick out the data points with 0 and 9
handled_data = []
handled_label = []
for i in range(len(train_y)):
    if train_y[i] == 0:
        handled_data.append(np.array(train_X[i, :], np.float32))
        handled_label.append(-1)
    elif train_y[i] == 9:
        handled_data.append(np.array(train_X[i, :], np.float32))
        handled_label.append(1)

handled_data = np.mat(handled_data)
handled_label = np.asarray(handled_label, dtype=np.float32)

pca = PCA(n_components=120, whiten=True)
pca.fit(handled_data)
handled_data = pca.transform(handled_data)
train_data, val_data, train_y, val_y = train_test_split(handled_data, handled_label, random_state=42, test_size=0.1)
estimator = SVM(C=1.0, gamma=10)
start = time.time()
print 'begin training'
estimator.fit(train_data[0:2000], train_y[0:2000])
print 'trianing done {} seconds'.format(time.time() - start)
print 'do prediction'
prediction = estimator.predict(val_data)
accuracy = accuracy_score(val_y, prediction)
print 'prediction score with {}'.format(accuracy)
