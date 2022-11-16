#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

int countNumberOfRice(cv::Mat src);

int main(int argc, char* argv[]){

cv::Mat src = cv::imread("/home/rovi2022/Documents/Machine-Vision/Reports/Report-04/rice.png", cv::IMREAD_GRAYSCALE);
cv::namedWindow("Input image",cv::WINDOW_FULLSCREEN);
cv::imshow("Input image", src);
//There are N-1 rice grains since one class is background
std::cout << "Number of objects: ";
std::cout << countNumberOfRice(src)<<std::endl;
std::cout << std::endl; 
}


int countNumberOfRice(cv::Mat src){
    cv::Mat resultImage = src.clone();
    cv::Mat kernel = getStructuringElement(cv::MorphShapes::MORPH_CROSS, cv::Size(3, 3), cv::Point(-1,-1));
    int iteration = 5; 
    cv::threshold(src, resultImage, 0, 255, cv::THRESH_BINARY|cv::THRESH_OTSU);
    cv::erode(resultImage, resultImage, kernel, cv::Point(-1,-1), iteration);
    cv::dilate(resultImage, resultImage, kernel, cv::Point(-1,-1), iteration);
    return cv::connectedComponents(resultImage, resultImage);
}
