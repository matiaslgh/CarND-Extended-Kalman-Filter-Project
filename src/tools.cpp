#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::cout;
using std::endl;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  // Initialize rmse vector
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  // Validate inputs
  if (estimations.size() == 0) {
    cout << "Error: Estimations size is zero" << endl;
    return rmse;
  }

  if (estimations.size() != ground_truth.size()){
    cout << "Error: ground_truth and estimations must have the same size" << endl;
    return rmse;
  }

  // Accumulate squared difference
  for (int i = 0; i < estimations.size(); i++){
    VectorXd diff = estimations[i] - ground_truth[i];
    diff = diff.array() * diff.array();
    rmse += diff;
  }

  rmse = rmse / estimations.size();
  rmse = rmse.array().sqrt();

  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3, 4);

  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  float d1 = px * px + py * py;
  float d2 = sqrt(d1);
  float d3 = d1*d2;

  if (fabs(d1) < 0.0001) {
    cout << "Error: Divison by zero." << endl;
    return Hj;
  }

  Hj << px / d2, py / d2, 0, 0,
        -py / d1, px / d1, 0, 0,
        py * ( vx * py - vy * px) / d3, px * (vy *px - vx * py) / d3, px / d2, py / d2;

  return Hj;
}
