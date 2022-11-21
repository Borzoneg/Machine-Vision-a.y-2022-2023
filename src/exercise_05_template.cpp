#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char **argv) {
  
  (void)argc;
  (void)argv;

  std::vector<cv::String> fileNames;
  cv::glob("../imgs/exercise_05_calibrationImages/Image*.png", fileNames, false);

  cv::Size patternSize(25 - 1, 18 - 1);
  std::vector<std::vector<cv::Point2f>> q(fileNames.size());

  // Detect feature points
  std::size_t i = 0;
  std::vector<cv::Point2f> corners; //this will be filled by the detected corners

  cv::TermCriteria termcrit(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,20,0.03);
  cv::Mat img = cv::imread("../imgs/chessboard.jpg");

  for (auto const &f : fileNames) {
    std::cout << std::string(f) << std::endl;
    
    // 1. Read in the image an call cv::findChessboardCorners()
    cv::findChessboardCorners(img, cv::Size(8,8), corners);
    // 2. Use cv::cornerSubPix() to refine the found corner detections
    cv::cornerSubPix(img, corners, patternSize, cv::Size(-1, -1), termcrit);      

    


    // Display
    cv::drawChessboardCorners(img, patternSize, q[i], success);
    cv::imshow("chessboard detection", img);
    cv::waitKey(0);

    i++;
  }

  std::vector<std::vector<cv::Point3f>> Q;
  // 3. Generate checkerboard (world) coordinates Q. The board has 25 x 18
  // fields with a size of 15x15mm

  cv::Matx33f K(cv::Matx33f::eye());  // intrinsic camera matrix
  cv::Vec<float, 5> k(0, 0, 0, 0, 0); // distortion coefficients

  std::vector<cv::Mat> rvecs, tvecs;
  std::vector<double> stdIntrinsics, stdExtrinsics, perViewErrors;
  int flags = cv::CALIB_FIX_ASPECT_RATIO;
  cv::Size frameSize(1440, 1080);

  std::cout << "Calibrating..." << std::endl;
  // 4. Call "float error = cv::calibrateCamera()" with the input coordinates
  // and output parameters as declared above...

  std::cout << "Reprojection error = " << error << "\nK =\n"
            << K << "\nk=\n"
            << k << std::endl;

  // Precompute lens correction interpolation
  cv::Mat mapX, mapY;
  cv::initUndistortRectifyMap(K, k, cv::Matx33f::eye(), K, frameSize, CV_32FC1,
                              mapX, mapY);

  // Show lens corrected images
  for (auto const &f : fileNames) {
    std::cout << std::string(f) << std::endl;

    cv::Mat img = cv::imread(f, cv::IMREAD_COLOR);

    cv::Mat imgUndistorted;
    // 5. Remap the image using the precomputed interpolation maps.

    // Display
    cv::imshow("undistorted image", imgUndistorted);
    cv::waitKey(0);
  }

  return 0;
}
