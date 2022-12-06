#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>

typedef struct gradientMap{
	cv::Mat gradientStrengthMap;
	cv::Mat gradientDirectionMap;
}gradientMap;

cv::Mat applyKernel(cv::Mat img, cv::Mat kernel);
cv::Mat paddingImg(cv::Mat img, int paddingSize);
gradientMap findGradientMap(cv::Mat img); 
cv::Mat applyCanny(cv::Mat, gradientMap);

int main(int argc, char **argv) {
    cv::Mat img = cv::imread("../inputFiles/imgs/legoHouse.jpg", cv::IMREAD_GRAYSCALE);
	//cv::Mat img = cv::imread("../inputFiles/imgs/test_gaussian.png", cv::IMREAD_GRAYSCALE);
	cv::Mat sobelHorizontalKernel = (cv::Mat_<float>(3,3) <<  1.0,  2.0,  1.0,
															  0.0,  0.0,  0.0,
															 -1.0, -2.0, -1.0);

	cv::Mat sobelVerticalKernel = (cv::Mat_<float>(3,3) <<   -1.0,  0.0,  1.0,
															 -2.0,  0.0,  2.0,
															 -1.0,  0.0,  1.0);
															 
	cv::Mat gaussian = (cv::Mat_<float>(5,5) <<   2.0,  4.0,  5.0,  4.0, 2.0,
												  4.0,  9.0, 12.0,  9.0, 4.0,
												  5.0, 12.0, 15.0, 12.0, 5.0,
												  4.0,  9.0, 12.0,  9.0, 4.0,
												  2.0,  4.0,  5.0,  4.0, 2.0);
	gaussian = gaussian * (1.0/159.0);	
	cv::Mat linearKernel(3,3, CV_32FC1, cv::Scalar(1.0/9.0));
																// risultato con Hsobel
	cv::Mat example = (cv::Mat_<uchar>(5,5) <<  0, 0, 0, 0, 0,  // -1, -3, -4, -3, -1, 
                                                0, 1, 1, 1, 0,  // -1, -3, -4, -3, -1,
                                                0, 1, 1, 1, 0,  //  0,  0,  0,  0,  0,
                                                0, 1, 1, 1, 0,  //  1,  3,  4,  3,  1,
                                                0, 0, 0, 0, 0   //  1,  3,  4,  3,  1
                                                );
	
	cv::Mat filtered = applyKernel(img, gaussian);							
	cv::convertScaleAbs(filtered, filtered);
	
	cv::Mat appliedV = applyKernel(img, sobelVerticalKernel);
	cv::Mat appliedH = applyKernel(img, sobelHorizontalKernel);
	cv::convertScaleAbs(appliedH, appliedH);
	cv::convertScaleAbs(appliedV, appliedV);

	cv::Mat appliedBothKo = applyKernel(appliedV, sobelHorizontalKernel);
	cv::convertScaleAbs(appliedBothKo, appliedBothKo);
	gradientMap appliedBothOk = findGradientMap(img);
	cv::convertScaleAbs(appliedBothOk.gradientStrengthMap, appliedBothOk.gradientStrengthMap);

	cv::Mat appliedCanny = img.clone();
	double lowTreshold = 0, ratio = 3;
	cv::Canny(img, appliedCanny, lowTreshold, lowTreshold*ratio);
	
	std::vector<cv::Vec3f> circles; // will hold the results of the detection
	cv::Mat houghLines;
	
	houghLines = img.clone();
	cv::HoughCircles(filtered, circles, cv::HOUGH_GRADIENT, 1, 40, 100, 30, 1, 50);
	for( size_t i = 0; i < circles.size(); i++ ){
		cv::Vec3i c = circles[i];
		cv::Point center = cv::Point(c[0], c[1]);
		// circle center
		cv::circle(houghLines, center, 1, cv::Scalar(0,100,100), 3, cv::LINE_AA);
		// circle outline
		int radius = c[2];
		circle(houghLines, center, radius, cv::Scalar(255,0,255), 3, cv::LINE_AA);
	}

	cv::imshow("Image", img); 
	cv::imshow("AppliedH", appliedH);
	cv::imshow("AppliedV", appliedV);
	cv::imshow("Applied both in wrong way", appliedBothKo);
	cv::imshow("Image filtered", filtered); 
	cv::imshow("Applied both properly", appliedBothOk.gradientStrengthMap);
	cv::imshow("Applied canny", appliedCanny);
	cv::imshow("Applied Hough", houghLines);	
	
	cv::waitKey(0);
	return 0;
}


gradientMap findGradientMap(cv::Mat img){
	int i, j;
	cv::Mat sobelHorizontalKernel = (cv::Mat_<float>(3,3) <<  1.0,  2.0,  1.0,
															  0.0,  0.0,  0.0,
															 -1.0, -2.0, -1.0);

	cv::Mat sobelVerticalKernel = (cv::Mat_<float>(3,3) <<   -1.0,  0.0,  1.0,
															 -2.0,  0.0,  2.0,
															 -1.0,  0.0,  1.0);

	cv::Mat mapV = applyKernel(img, sobelVerticalKernel);
	cv::Mat mapH = applyKernel(img, sobelHorizontalKernel);

	gradientMap mapVH;
	mapVH.gradientStrengthMap = mapV.clone();
	mapVH.gradientDirectionMap = mapV.clone();

	for(i=0; i<mapH.rows; i++){
		for(j=0; j<mapH.cols; j++){
			mapVH.gradientStrengthMap.at<float>(i,j) = sqrt(pow(mapH.at<float>(i,j), 2) + pow(mapV.at<float>(i,j), 2));
			mapVH.gradientDirectionMap.at<float>(i,j) = atan(mapV.at<float>(i,j) / mapH.at<float>(i,j));
		}
	}
	return mapVH;
}


cv::Mat applyKernel(cv::Mat img, cv::Mat kernel){
	int i, j, k, l, corr = kernel.rows/2, tmp=0;
	float sum = 0;
    // the size of the kernel divided by 2 will be our correction factor to scale the padded image and to offset by the right amount when applying the kernel
    cv::Mat applied (img.rows, img.cols, CV_32FC1);
    cv::Mat paddedImg = paddingImg(img, corr);
	for(i=corr; i<paddedImg.rows-corr; i++){ 
		for(j=corr; j<paddedImg.cols-corr; j++){
			for(k=-corr; k<=corr; k++){
				for(l=-corr; l<=corr; l++){
					sum += (kernel.at<float>(k+corr, l+corr) * paddedImg.at<uchar>(i+k, j+l));   
				}
			}
            applied.at<float>(i-corr, j-corr) = sum;
			sum = 0;
		}
	} 
	return applied;
}


cv::Mat paddingImg(cv::Mat img, int paddingSize){
    cv::Mat paddedImg(img.rows+paddingSize*2, img.cols+paddingSize*2, img.type(), cv::Scalar(0.0));
    int i, j;
    for(i=0; i<img.rows; i++){
        for(j=0; j<img.cols; j++){
            paddedImg.at<uchar>(i+paddingSize, j+paddingSize) = img.at<uchar>(i, j);
        }
    }
    return paddedImg;
}
