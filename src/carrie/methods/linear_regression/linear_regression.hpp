#ifndef CARRIE_LINEAR_REGRESSION_HPP_
#define CARRIE_LINEAR_REGRESSION_HPP_
#include "linear_base.hpp"
namespace carrie {
namespace regression /**regression method**/ {
/***
 * this is the ordinary linear regression method
 **/
class LinearRegression : public LinearBase {
public:
    LinearRegression() {}
    ~LinearRegression() {}
    void train(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    Eigen::MatrixXd predict(const Eigen::MatrixXd& X);

};//class linear regression

}//namespace regression
}//namespace carrie

