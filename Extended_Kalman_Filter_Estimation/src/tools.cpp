#include "tools.h"
#include <iostream>

using std::cout;
using std::endl;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

//constructor
Tools::Tools() {}
//destructor
Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  //integrity check: 1) estimation vector size larger than 0; 2) ground truth vector size larger than 0; 3) estimation vector size same with ground_truth vector size
  if (estimations.size()==0 || ground_truth.size()==0 || estimations.size()!=ground_truth.size()){
    cout << "Check your estimation or ground truth data set." << endl;
    return rmse;
  }
  //calculate rmse
  //1. calculation of squared difference b/w estimation and ground truth
  for (unsigned int i=0; i<estimations.size(); ++i){
    VectorXd difference = estimations[i]-ground_truth[i];

    difference = difference.cwiseProduct(difference);
    rmse += difference;
  }

  //2. mean of estimations
  rmse = rmse/estimations.size();

  //3. calculation of squared root
  rmse = rmse.array().sqrt();

  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  //initialise jacobian matrix h - 3 (rho, theta, rho dot) by 4 (x, y, vx, vy)
  MatrixXd Hj(3, 4);

  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  float x2y2 = px*px + py*py;
  float rx2y2 = sqrt(x2y2);
  float x32y32 = x2y2*rx2y2;

  //integrity check that denominator is not zero
  if (fabs(x2y2)<0.0001){
    cout << "Division by zero error - Jacobian denominator is zero." << endl;
    return Hj;
  }

  //calculation of jacobian matrix - partial derivatives
  Hj << (px/rx2y2), (py/rx2y2), 0, 0,
       -(py/x2y2), (px/x2y2), 0, 0,
       py*(vx*py - vy*px)/x32y32, px*(vy*px - vx*py)/x32y32, px/rx2y2, py/rx2y2;
  return Hj;
}
