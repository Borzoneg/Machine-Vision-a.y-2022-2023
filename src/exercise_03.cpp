#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

void showHist(cv::Mat grayImage, std::string namedWindow);


int main(int argc, char* argv[]){
    int i, j, valuableBean = 0;
    cv::Mat img = cv::imread("../imgs/TrinityCampanile3.jpg", cv::IMREAD_GRAYSCALE);
    cv::imshow("img", img);
    showHist(img, "img histogram");
    std::vector<float> repetition(255, 0.0);
    float tmp, acc = 0, acc2 = 0;

    for(i=0; i<img.rows; i++){
        for(j=0; j<img.cols; j++){
            repetition[img.at<uchar>(i, j)]++;
        }
    }
    acc = 0;
    for(i=0; i<img.rows; i++){
        for(j=0; j<img.cols; j++){
            tmp = (float)repetition[img.at<uchar>(i, j)] / (float)(img.rows * img.cols);
            img.at<uchar>(i, j) = tmp;
            acc2 += tmp;
            //std::cout<<tmp<<std::endl;
        }
    }

    std::cout<<"Should sum to 1: "<<acc2<<std::endl;
    cv::imshow("img equalized", img);
    showHist(img, "img equalized histogram");
    //cv::waitKey(0);
    return 0;
}


void showHist(cv::Mat grayImage, std::string windowName){
    CV_Assert(grayImage.type()==CV_8UC1);
      int histSize = 256;
      float range[] = { 0, 256 } ;
      const float* histRange = { range };
      bool uniform = true;
      bool accumulate = false;

      cv::Mat hist;
      cv::calcHist( &grayImage, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );

      //Initialize histogram image
      int hist_w = 512; int hist_h = 400;
      int bin_w = cvRound( (double) hist_w/histSize );
      cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );
      
      // Normalize the result to [ 0, histImage.rows ]
      cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );

      // Draw line between each (bin,value)
      for( int i = 1; i < histSize; i++ ){
          cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
                           cv::Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
                           cv::Scalar( 255, 0, 0), 2, 8, 0  );
      }

      // Display
      cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE );
      cv::imshow(windowName, histImage );
}
