#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
  x_ = F_ * x_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd z_pred = H_ * x_;
  DoUpdate(z - z_pred);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  float theta = atan2(x_(1), x_(0));
  float rho = sqrt(x_(0) * x_(0) + x_(1) * x_(1));
  float rho_dot = (x_(0) * x_(2) + x_(1) * x_(3)) / rho;
  
  VectorXd h = VectorXd(3);
  h << rho, theta, rho_dot;

  DoUpdate(z - h);
}

void KalmanFilter::DoUpdate(const VectorXd &error){
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K =  P_ * Ht * Si;
  
  // New state
  x_ = x_ + (K * error);
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - K * H_) * P_;
}