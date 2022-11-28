#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


void rearrange(cv::Mat toBeArranged);
cv::Mat rotate(double angle, cv::Mat toRotate);


int main(int argc, char **argv) {
    cv::Mat img = cv::imread("../imgs/rotatedText.png", cv::IMREAD_GRAYSCALE);
    
    cv::Mat padded;
    int m = cv::getOptimalDFTSize(img.rows);
    int n = cv::getOptimalDFTSize(img.cols); 
    copyMakeBorder(img, padded, (m - img.rows)/2, (m - img.rows)/2, (n - img.cols)/2, (n - img.cols)/2, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat fourier;
    cv::merge(planes, 2, fourier);  
    cv::dft(fourier, fourier);
    
    cv::split(fourier, planes);
    cv::magnitude(planes[0], planes[1], planes[0]); // ???
    cv::Mat amplitude = planes[0];
    cv::Mat phase = planes[1];

    // add 1 to not run into problem with the log and go to log scale
    amplitude += cv::Scalar::all(1);                    
    log(amplitude, amplitude);
    phase += cv::Scalar::all(1);                    
    log(phase, phase);

    // crop the spectrum, if it has an odd number of rows or columns
    amplitude = amplitude(cv::Rect(0, 0, amplitude.cols & -2, amplitude.rows & -2)); // &: bitwise and
    phase = phase(cv::Rect(0, 0, phase.cols & -2, phase.rows & -2)); // &: bitwise and
    

    // rearrange the quadrants of Fourier image  so that the origin is at the image center and normalize it for visualization
    rearrange(amplitude);
    normalize(amplitude, amplitude, 0, 1, cv::NORM_MINMAX);
    rearrange(phase);
    normalize(phase, phase, 0, 1, cv::NORM_MINMAX); 

    cv::imshow("Image", img);
    cv::imshow("Amplitude", amplitude);
    cv::imshow("Phase", phase);
    cv::waitKey(0);
    
    img = rotate(-10, img);
    cv::imshow("Image rotated", img);
    cv::waitKey(0);
    
    return 0;
}


void rearrange(cv::Mat toBeArranged){
    int cx = toBeArranged.cols/2;
    int cy = toBeArranged.rows/2;
    cv::Mat q0(toBeArranged, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    cv::Mat q1(toBeArranged, cv::Rect(cx, 0, cx, cy));  // Top-Right
    cv::Mat q2(toBeArranged, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    cv::Mat q3(toBeArranged, cv::Rect(cx, cy, cx, cy)); // Bottom-Right
    cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
}


cv::Mat rotate(double angle, cv::Mat toRotate){
    // get rotation matrix for rotating the image around its center in pixel coordinates
    cv::Point2f center((toRotate.cols-1)/2.0, (toRotate.rows-1)/2.0);
    cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
    // determine bounding rectangle, center not relevant
    cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), toRotate.size(), angle).boundingRect2f();
    // adjust transformation matrix
    rot.at<double>(0,2) += bbox.width/2.0 - toRotate.cols/2.0;
    rot.at<double>(1,2) += bbox.height/2.0 - toRotate.rows/2.0;
    cv::Mat dst;
    cv::warpAffine(toRotate, dst, rot, bbox.size());
    return dst;
}
