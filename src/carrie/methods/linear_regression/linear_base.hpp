#ifndef CARRIE_LINEAR_BASE_HPP_
#define CARRIE_LINEAR_BASE_HPP_
#include "carrie/prereqs.hpp"
#include "carrie/util/log.hpp"
namespace carrie {
namespace regression /**regression method**/{
/**
 * This is the base class for all the linear regression method. Including
 * the ordinary linear regression, ridge, lasso, bayesian regression, polynomial regression
 **/
class LinearBase {
public:
    virtual LinearBase() { instanced_ = false; }
    virtual ~LinearBase() {}
    virtual void train(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, bool intercept=true) = 0;
    virtual Eigen::MatrixXd predict(const Eigen::MatrixXd& X) = 0;

    virtual Eigen::VectorXd get_params() const {
        if (!instanced_) {
            LOG(WARN) << "params has not been instanced!";
            return NULL;
        }
        return params_;
    }

    virtual void set_params(const Eigen::VectorXd& params) {
        if (instanced_) {
            if(params.size() != params_.size()) {
                LOG(WARN) << "params size not agree with the original param size";
            }
        }
        params_ = params;
    }
protected:
    Eigen::VectorXd params_;
    bool instanced_;
    bool intercept_;
};// end of the LinearBase

}//namespace regression
}//namespace carrie
#endif
