#include "opencv2/opencv.hpp"
#include <array>
#include <iostream>
#include <algorithm>

int main(int argc, char *argv[]){
    cv::VideoCapture cap ("../inputFiles/PETS2000.avi");
    std::vector<cv::Mat> frames, medianBgImgs;
    cv::Mat tmp;
    int i, j, k, a, b, c;

    cap>>tmp;
    while(!tmp.empty()){ // while there are frame
        cv::cvtColor(tmp, tmp, cv::COLOR_BGR2GRAY); // convert it to grayscale
        frames.push_back(tmp); // add it to the vector we will use
        // cv::imshow("video", tmp); // show them
        // cv::waitKey(1);
        cap>>tmp; // get a new frame
    }
    for(i=2; i<frames.size(); i++){ // for each frame starting from the third
        tmp = frames[i].clone(); 
        for(j=0; j<frames[i].rows; j++){ // we look for the median of the last three frames
            for(k=0; k<frames[i].cols; k++){
                a = frames[i].at<uchar>(j, k);
                b = frames[i-1].at<uchar>(j, k);
                c = frames[i-2].at<uchar>(j, k);
                tmp.at<uchar>(j, k) = std::max(std::min(a,b), std::min(std::max(a,b),c));
            }
        }
        medianBgImgs.push_back(tmp); // and push it in the median imgs vector
    }
    frames.erase(frames.begin()); // delete the first two because we haven't computed the median for them
    frames.erase(frames.begin());

    for(i=0; i<frames.size(); i++){ // visualize the video again, the medians and the result
        cv::imshow("video", frames[i]);
        cv::absdiff(frames[i], medianBgImgs[i], tmp);
        cv::threshold(tmp, tmp, 30, 255, 0); // to filter out some errors from the unprecision fo the cameras
        cv::imshow("backgroundsub", tmp);
        cv::imshow("median", medianBgImgs[i]);
        cv::waitKey(10);
    }
    return 0;
}