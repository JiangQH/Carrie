#ifndef CARRIE_BASE_REGRESSOR_HPP_
#define CARRIE_BASE_REGRESSOR_HPP_
#include <carrie/prereqs.hpp>
namespace carrie {
namespace regression {
class BaseRegressor {
public:
    virtual BaseRegressor(const Eigen::MatrixXd& X,
                  const Eigen::VectorXd& y) {};
    virtual BaseRegressor() {};
    virtual ~BaseRegressor(){};
    virtual void train(const Eigen::MatrixXd& X,
               const Eigen::VectorXd& y)=0;

    virtual const Eigen::VectorXd& predict(const Eigen::MatrixXd& test_X) const=0;

    virtual void set_params(const Eigen::VectorXd& params) {
        params_ = params;
    }

    virtual const Eigen::VectorXd& get_params() const {
        return params_;
    }

protected:
    Eigen::VectorXd params_;
    bool instanced_; 

};//end of class base regressor


}// end of namespace regression
}// end of namespace carrie
#endif // end of the define
