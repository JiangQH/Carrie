#ifndef CARRIE_BASE_REGRESSOR_HPP_
#define CARRIE_BASE_REGRESSOR_HPP_
#include <carrie/prereqs.hpp>
namespace carrie {
namespace regression {
class BaseRegressor {
public:
    virtual BaseRegressor(const Eigen::MatrixXd& X,
                  const Eigen::VectorXd& y)=0;
    virtual BaseRegressor()=0;
    virtual ~BaseRegressor()=0;
    virtual void train(const Eigen::MatrixXd& X,
               const Eigen::VectorXd& y)=0;

    virtual Eigen::VectorXd predict(const Eigen::MatrixXd& test_X) const=0;

    virtual void set_params(const Eigen::VectorXd& params)=0;

    virtual const Eigen::VectorXd& get_params() const=0;

private:
    Eigen::VectorXd params_;

};//end of class base regressor


}// end of namespace regression
}// end of namespace carrie
#endif // end of the define
