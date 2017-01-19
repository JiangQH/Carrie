//
// Created by jqh on 16-12-27.
//
#include "logistic_regression.hpp"
namespace carrie {

// the training function, this will fill the params
template <typename MatType>
void LogisticRegression<MatType>::fit(const MatType &X, const MatType &y, double lr, max_eps) {

}

// the prediction, using the params
template <typename MatType>
MatType LogisticRegression<MatType>::predict(const MatType &test_X) {

}


// loss function of logistic regression
template <typename MatType>
MatType LogisticRegression<MatType>::loss(const MatType &X, const MatType &y) {

}

// get the weights
template <typename MatType>
const MatType& LogisticRegression<MatType>::getWeights() {

}


// the sigmoid function, forming the hypythsis
template <typename MatType>
MatType LogisticRegression<MatType>::sigmoid(const MatType &z) {

}

// gradient of the loss
template <typename MatType>
MatType LogisticRegression<MatType>::loss_grad(const MatType &X, const MatType &y) {

}

}// end of carrie


