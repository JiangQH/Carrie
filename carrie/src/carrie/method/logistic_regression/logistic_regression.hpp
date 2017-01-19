//
// Created by jqh on 16-12-27.
//
#ifndef CARRIE_LOGISTIC_REGRESSION_HPP
#define CARRIE_LOGISTIC_REGRESSION_HPP
#include <vector>
#include "carrie/core.hpp"

namespace carrie {
template <typename MatType = Eigen::MatrixXd>
class LogisticRegression {
public:
    void fit(const MatType& X, const MatType& y, double lr=0.01, max_eps=200);
    MatType predict(const MatType& test_X);
    MatType loss(const MatType& X, const MatType& y);
    const MatType& getWeights();
private:
    MatType sigmoid(const MatType& z);
    MatType loss_grad(const MatType& X, const MatType& y);

    MatType _params;


};// end of class
}// end of carrie
#endif //CARRIE_LOGISTIC_REGRESSION_HPP
