#include "kalman_filter.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in; //state input vector
  P_ = P_in; //state covariance
  F_ = F_in; //state transition matrix
  H_ = H_in; //measurement transiation matrix
  R_ = R_in; //measurement covariance matrix
  Q_ = Q_in; //process covariance matrix
}

void KalmanFilter::Predict() {
  //1. predict step - overall, predict new state vector and covariance (errors) of this prediction
  //state transition matrix*state vector + noise with gaussian distribution (omitted as it has a mean around 0)
  x_ = F_*x_;
  //F is state transition matrix with delta time
  MatrixXd Ft = F_.transpose();
 //state covariance matrix P representing error in process, in addition to uncertainty from acceleration (process covariance Q)
  P_ = F_* P_ * Ft + Q_;
}

void KalmanFilter::UpdateBase(const VectorXd &y){
  //total covariance error including process covariance error and measurement covariance error
  //H_ measurement transition matrix - if radar, jacobian matrix passed
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;

  //2. calculate Kalman Gain - state covariance over total error calculating weight
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;

  //3. weigh with Kalman Gain and update state vector 
  //i.e. apply kalman gain weight (if process error is large or measurement error is small, K is large) and update state vector
  x_ = x_ + (K*y);

  //3. weigh with Kalman Gain and update process covariance
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - K * H_ ) * P_;
}

void KalmanFilter::Update(const VectorXd &z) {
  //transition state vector to the measurement space of a sensor
  VectorXd z_pred = H_ * x_;

  //calculate error
  VectorXd y = z - z_pred;

  //calculate kalman gain, weight and update state vector and process covariance
  UpdateBase(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  //decompose state vector
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);
  
  //precompute denominator
  float dnom = px*px + py*py;
  dnom = sqrt(dnom);
  //integrity check for denominators before conversion
  if (fabs(dnom)<0.0001){
    px += 0.0001;
    py += 0.0001;
  }

  //convert from cartesian coordinates to polar coordinates
  float rho = dnom;
  float theta = atan2(py, px);
  float rhoDot = (px*vx+py*vy)/dnom;
  
  VectorXd Hj = VectorXd(3);
  Hj << rho, theta, rhoDot;

  //calculate error
  VectorXd y = z - Hj;

  //normalise theta so that it is between -pi and +pi
  float eTheta = y(1);
  while (eTheta>M_PI){
    eTheta -= 2*M_PI;
  }
  while (eTheta<-M_PI){
    eTheta += 2*M_PI;
  }
  y(1) = eTheta;
  
  //calculate kalman gain, weight and update state vector and process covariance
  UpdateBase(y);
}
