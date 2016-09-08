/*************************************************************************
	> File Name: linear_regression.cpp
	> Author: 
	> Mail: 
	> Created Time: 2016年09月08日 星期四 10时13分51秒
 ************************************************************************/
// Todo --- define self assert to replace the if judge condition
#include "linear_regression.hpp"
using namespace carrie;
using namespace carrie::regression;

void LinearRegression::train(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, bool intercept) {
    /***
     * X: sparse matrix of shape[m_samples, n_features]
     * y: array of shape [m_samples]
     * intercept: whether to add the intercept to the data.
     * train the params, params_: the parameter, which is [n_features]
     **/
   // first check m_samples equals or not
    if (X.rows() != y.size()) {
        LOG(ERROR) << "the rows of X and y should be same.";
        return;
    }
   this->intercept_ = intercept;
   // if intercept, then should add a first cloumn to the X, to make the w0 be included
   const size_t m_rows = X.rows();
   const size_t n_cols = X.cols();
   if (this->intercept_) {
       Eigen::VectorXd inter = Eigen::VectorXd::Constant(m_rows, 1.0);
       Eigen::MatrixXd x_copy(m_rows, n_cols+1);
       x_copy << inter, X;
   }
   else {
       Eigen::MatrixXd x_copy = X;
   }

   // now solve the y = Xw problem to minimize the square loss ||Xw - y||^2 loss function
   // eigen provides 3 methods to solve this problem.
   // the svd decomposition, QR decomposition, and normal equations.
   // here use the QR decomposition to compromise between speed and accuracy
   this->params_ = X.colPivHouseholderQr().solve(y);
   this->instanced_ = true;
}


Eigen::MatrixXd LinearRegression::predict(const Eigen::MatrixXd& X) {
    /**
     * X: [m_samples, n_features]
     * params_ : [n_features]
     **/ 
    // if it is trained?
    if (!this->instanced_) {
        LOG(ERROR) << "the model has not been trained, train it first.";
        return NULL;
    }
    const size_t m_rows = X.rows();
    const size_t n_cols = X.cols();
    if (this->intercept_) {
        Eigen::VectorXd inter = Eigen::VectorXd::Constant(m_rows, 1.0);
        Eigen::MatrixXd x_copy(m_rows, n_cols + 1);
        x_copy << inter, X;
    }
    else {
        Eigen::MatrixXd x_copy = X;
    }
    // y = XW, make sure the dimension agree
    if (x_copy.cols() != this->params_.size()) {
        LOG(ERROR) << "the input x's feature dimension is not agree with the trained params";
        return NULL;
    }
    return x_copy * this->params_;
}




 
