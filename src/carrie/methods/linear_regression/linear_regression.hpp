#ifndef LINEAR_REGRESSION_HPP_
#define LINEAR_REGRESSION_HPP_
#include "linear_base.hpp"
namespace carrie {
namespace regression /**regression method**/ {
class LinearRegression : public LinearBase {
public:
    LinearRegression() {}
    ~LinearRegression() {}
    void train(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    Eigen::MatrixXd predict(const Eigen::MatrixXd& X);
    Eigen::VectorXd get_params();
    void set_params(const Eigen::VectorXd& params);
private:

};//class linear regression

}//namespace regression
}//namespace carrie
#endif
