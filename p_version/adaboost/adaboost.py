import numpy as np
import tools

def classify(data_mat, feature_ind, thresh_hold, judge):
    m = data_mat.shape[0]
    prediction = np.ones((m, 1))
    if judge == 'gt':
        prediction[data_mat[:, feature_ind] >= thresh_hold] = -1
    elif judge == 'lt':
        prediction[data_mat[:, feature_ind] < thresh_hold] = -1
    else:
        print 'not defined'
    return prediction


def buildTree(data_attr, labels, D):
    # data_mat is (m, n)
    # labels is (m, 1)
    data_mat = np.mat(data_attr)
    labels = np.mat(labels).T
    m, n = data_mat.shape

    # n is the feature
    num_steps = 10
    best_tree = {}
    min_error = np.inf
    best_predict = np.ones((m, 1))
    for feature_ind in range(n):
        # for every feature, compute the minium and max and stepsize
        min_feature = np.min(data_mat[:, feature_ind])
        max_feature = np.max(data_mat[:, feature_ind])
        step_size = (max_feature - min_feature) * 1.0 / num_steps

        #for the step
        for step in range(-1, num_steps+1):
            thresh_hold = min_feature + step * step_size

            #classify the data using this threshhold
            for judge in ['gt', 'lt']:
                predict_label = classify(data_mat, feature_ind, thresh_hold, judge)
                error_count = np.mat(np.zeros((m, 1)))
                error_count[predict_label != labels] = 1
                error = error_count.T * D
               # print "split: ind {}, thresh {}, judge {}, error {}".format(feature_ind, thresh_hold, judge, error)
                if error < min_error:
                    min_error = error
                    best_predict = predict_label.copy()
                    best_tree['dim'] = feature_ind
                    best_tree['thresh_hold'] = thresh_hold
                    best_tree['judge'] = judge

    return best_tree, min_error, best_predict

def trainAdaBoost(data_attr, labels, num_iter=40):
    # init D
    m, n = np.shape(data_attr)
    D = np.mat(np.ones((m, 1)) / m)
    classifiers = []
    estimations = np.mat(np.zeros((m, 1)))
    for i in range(num_iter):
        # using the current D to train a weak classifier
        tree, error, predict = buildTree(data_attr, labels, D)
       # print "D is {}".format(D.T)
        # update the alpha
        alpha = 0.5 * np.log((1 - error) / max(error, 1e-16)) # here is a trick not to divide zero
        alpha = np.asscalar(alpha)
        tree['alpha'] = alpha
        # update the D
        label_mat = np.diag(labels)
        exps = np.exp(-alpha * label_mat * np.mat(predict))
        z = D.T * exps
        D = np.multiply(D, exps) / z
        # store the info
        classifiers.append(tree)
        estimations += alpha * predict
        total_error = np.multiply(np.sign(estimations) != np.mat(labels).T, np.ones((m, 1)))
        total_error = total_error.sum() / m
       # print "total error is {}".format(total_error)
        if total_error < 1e-3:
            break
    return classifiers


def predictAdaBoost(data_attr, classifiers):
    data = np.mat(data_attr)
    m = data.shape[0]
    predictions = np.mat(np.zeros((m, 1)))
    for i in range(len(classifiers)):
        predict = classify(data, classifiers[i]['dim'],
                 classifiers[i]['thresh_hold'],
                 classifiers[i]['judge'])
        predictions += classifiers[i]['alpha'] * predict
        #print predictions
    return np.sign(predictions)


data_attr, labels = tools.loadDataSet('../data/horseColicTraining2.txt')
classifiers = trainAdaBoost(data_attr, labels, num_iter=20)
test_attr, test_labels = tools.loadDataSet('../data/horseColicTest2.txt')
prediction = predictAdaBoost(test_attr, classifiers)
# get the error rate
error_rate = np.mat(np.ones((67, 1)))
error_rate = error_rate[prediction != np.mat(test_labels).T].sum() / np.shape(test_labels)[0]

print error_rate


