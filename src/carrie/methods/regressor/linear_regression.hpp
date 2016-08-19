#ifndef CARRIE_LINEAR_REGRESSION_HPP_
#define CARRIE_LINEAR_REGRESSION_HPP_
#include <base_regressor.hpp>
namespace carrie {
namespace regression {
class LinearRegression::public BaseRegressor {
public:
    explicit LinearRegression(const Eigen::MatrixXd& X,
            const Eigen::VectorXd& y);
    explicit LinearRegression();
    
    ~LinearRegression();

    void train(const Eigen::MatrixXd& X,
            const Eigen::VectorXd& y);

    const Eigen::VectorXd& predict(const Eigen::MatrixXd& test_X) const;

};// class linear regression
};// namespace regression
}; // namespace carrie
#endif
