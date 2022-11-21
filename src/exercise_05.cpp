#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char **argv) {
	std::vector<cv::String> fileNames;
	cv::glob("../imgs/exercise_05_calibrationImages/Image*.png", fileNames, false);
	
    //cv::Size patternSize(cellInRow - 1, cellInRow - 1);
    cv::Size patternSize(25 - 1, 18 - 1);
    cv::Size windowSize(11, 11); // I din't understand why 11,11
    
	std::vector<std::vector<cv::Point2f>> q;
	// Detect feature points
	std::size_t i = 0;

	
	cv::TermCriteria termcrit(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,20,0.03);
    bool success;
    cv::Mat img;

	for (auto const &f : fileNames) {		
		// 1. Read in the image an call cv::findChessboardCorners()
        img = cv::imread(f, cv::IMREAD_GRAYSCALE);
        std::vector<cv::Point2f> corners; //this will be filled by the detected corners
        success = cv::findChessboardCorners(img, patternSize, corners);
        if(!success)
            break;
        q.push_back(corners);
		// 2. Use cv::cornerSubPix() to refine the found corner detections
		cv::cornerSubPix(img, corners, windowSize, cv::Size(-1, -1), termcrit);
		// Display
		cv::drawChessboardCorners(img, patternSize, q[i], success);
		i++;
	}

	std::vector<std::vector<cv::Point3f>> Q;
	for(i=0; i<q.size(); i++){
		std::vector<cv::Point3f> corners3D;
		for(int j=1; j<25; j++){
			for(int k=1; k<18; k++){
				corners3D.push_back(cv::Point3f(15*j,15*k,0));
			}
		}
		Q.push_back(corners3D);
	}
	std::cout<<Q.size()<<"   "<<q.size();
	// 3. Generate checkerboard (world) coordinates Q. The board has 25 x 18
	// fields with a size of 15x15mm

	cv::Matx33f K(cv::Matx33f::eye());	// intrinsic camera matrix
	cv::Vec<float, 5> k(0, 0, 0, 0, 0); // distortion coefficients
  
	std::vector<cv::Mat> rvecs, tvecs; // rotation and translation
	std::vector<double> stdIntrinsics, stdExtrinsics, perViewErrors;
	int flags = cv::CALIB_FIX_ASPECT_RATIO;
	cv::Size frameSize(1440, 1080);

	std::cout << "Calibrating..." << std::endl;
    
	// 4. Call "float error = cv::calibrateCamera()" with the input coordinates
	// and output parameters as declared above...
    float error = cv::calibrateCamera(Q, q, frameSize, K, k, rvecs, tvecs);
	std::cout << "Reprojection error = " << error << "\nK =\n"
						<< K << "\nk=\n"
						<< k << std::endl;

	// Precompute lens correction interpolation
	cv::Mat mapX, mapY;
	cv::initUndistortRectifyMap(K, k, cv::Matx33f::eye(), K, frameSize, CV_32FC1, mapX, mapY);

	// Show lens corrected images
	for (auto const &f : fileNames) {
		cv::Mat img = cv::imread(f, cv::IMREAD_COLOR);

		cv::Mat imgUndistorted;
		// 5. Remap the image using the precomputed interpolation maps.
		cv::remap(img, imgUndistorted, mapX, mapY, cv::INTER_LINEAR);
		// Display
		cv::imshow("distorted image", img);
		cv::waitKey(0);
		cv::imshow("undistorted image", imgUndistorted);
		cv::waitKey(0);
	}

	return 0;
}
