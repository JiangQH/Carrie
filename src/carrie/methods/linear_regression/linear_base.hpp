#ifndef CARRIE_LINEAR_BASE_HPP_
#define CARRIE_LINEAR_BASE_HPP_
#include "carrie/prereqs.hpp"

namespace carrie {
namespace regression /**regression method**/{
/**
 * This is the base class for all the linear regression method. Including
 * the ordinary linear regression, ridge, lasso, bayesian regression, polynomial regression
 **/
class LinearBase {
public:
    virtual LinearBase() {}
    virtual ~LinearBase() {}
    virtual void train(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) = 0;
    virtual Eigen::MatrixXd predict(const Eigen::MatrixXd& X) = 0;
    virtual Eigen::VectorXd get_params() = 0;
    virtual void set_params(const Eigen::VectorXd& params) = 0;

};// end of the LinearBase

}//namespace regression
}//namespace carrie
#endif
