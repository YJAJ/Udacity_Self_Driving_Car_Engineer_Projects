#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2); //laser measurement covariance matrix
  R_radar_ = MatrixXd(3, 3); //radar measurement covariance matrix
  H_laser_ = MatrixXd(2, 4); //measurement transition matrix for laser
  Hj_ = MatrixXd(3, 4); //measurement transition matrix for radar - jacobian matrix for non-linear to linear

  //measurement covariance matrix - laser (uncertainty in measurement)
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar (uncertainty in measurement)
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  //H_laser (non-linear func) for using px and py only
  H_laser_ << 1, 0, 0, 0,
             0, 1, 0, 0;

  //Hj Jacobian matrix for using rho, phi, rhoDot
  Hj_ << 1, 1, 0, 0,
         1, 1, 0, 0,
         1, 1, 1, 1; 

  //state transition matrix - (px, vx), (py, vy), vx, vy
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1;

  //state covariance matrix
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1000, 0,
            0, 0, 0, 1000;
  //process covariance matrix (uncertainty in process) is initialised in prediction with delta time
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) {
    // first measurement
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      //rho, phi, rho dot
      float rho = measurement_pack.raw_measurements_(0);
      float theta = measurement_pack.raw_measurements_(1);
      float rhoDot = measurement_pack.raw_measurements_(2);

      ekf_.x_(0) = rho * cos(theta);
      ekf_.x_(1) = rho * sin(theta);
      ekf_.x_(2) = rhoDot * cos(theta);
      ekf_.x_(3) = rhoDot * sin(theta);
      cout << "EKF Radar Initialisation Done." << endl;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      //x
      ekf_.x_(0) = measurement_pack.raw_measurements_(0);
      //y
      ekf_.x_(1) = measurement_pack.raw_measurements_(1);
      cout << "EKF Laser Initialisation Done." << endl;
    }

    //it is initialised now
    is_initialized_ = true;
    //start time
    previous_timestamp_ = measurement_pack.timestamp_;

    return;
  }

  //1. predict step

  //determine dt and reset current time to previous time
  float dt = (measurement_pack.timestamp_ - previous_timestamp_)/1000000.0;
  //make current to previous
  previous_timestamp_ = measurement_pack.timestamp_;

  //state transition matrix multiply dt - update transition matrix
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;
  
  //square, cube, ^4
  float dt2 = dt * dt;
  float dt3 = dt2 * dt;
  float dt4 = dt3 * dt;

  //process noise acceleration - 9 is provided by the instruction
  float acc_x = 9;
  float acc_y = 9;

  //process covariance matrix Q
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << dt4/4*acc_x, 0, dt3/2*acc_x, 0,
            0, dt4/4*acc_y, 0, dt3/2*acc_y,
            dt3/2*acc_x, 0, dt2*acc_x, 0,
            0, dt3/2*acc_y, 0, dt2*acc_y;
  
  //predict with all set up
  ekf_.Predict();

  //2 and 3. calculate Kalman Gain and update step
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    Tools tool;
    Hj_ = tool.CalculateJacobian(ekf_.x_);
    ekf_.H_ = Hj_; //measurement transition matrix
    ekf_.R_ = R_radar_; //measurement covariance for radar
    //update with extended kalman filter process
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
    ekf_.H_ = H_laser_; //measurement transition matrix
    ekf_.R_ = R_laser_; //measurement covariance for laser
    //update with kalman filter process
    ekf_.Update(measurement_pack.raw_measurements_);
  }
}