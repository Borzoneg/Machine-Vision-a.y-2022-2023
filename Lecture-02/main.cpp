#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
//#include <collection.h>

void linearFilter(cv::Mat src, cv::Mat k, cv::Mat &output);
void medianFilter(cv::Mat src, cv::Mat k, cv::Mat &output);

int main(int argc, char* argv[]){
//Load image as grayscale
    if(argc != 2){
        std::cout << "Usage: ./main <imagefile.jpg/png>"<< std::endl;
        return -1;
    }


    std::string filename = argv[1];
    cv::Mat src = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    cv::namedWindow("src");
    cv::imshow("src",src);
    CV_Assert(src.type() == CV_8UC1);
    
    //Create uniform 3x3 kernel
    cv::Mat linearKernel(3,3, CV_32FC1, cv::Scalar(1.0/9.0));
    CV_Assert(linearKernel.type() == CV_32FC1);

    //Apply linear filter
    cv::Mat linearOutput;
    // linearFilter(src, linearKernel, linearOutput);
    // cv::namedWindow("Linear filter linearOutput");
    // cv::imshow("Linear filter linearOutput",linearOutput);

    //Create cross kernel with 1
    cv::Mat medianKernel = (cv::Mat_<float>(3,3) << 0,1,0,1,1,1,0,1,0);
    CV_Assert(medianKernel.type() == CV_32FC1);

    //Apply median filter
    cv::Mat medianOutput;
    medianFilter(src, medianKernel, medianOutput);
    cv::namedWindow("Median filter medianOutput");
    cv::imshow("Median filter medianOutput",medianOutput);

    cv::Mat medianOutput2;
    medianFilter(medianOutput, medianKernel, medianOutput2);
    cv::namedWindow("Median filter medianOutput2");
    cv::imshow("Median filter medianOutput2",medianOutput2);

    cv::Mat medianOutput3;
    medianFilter(medianOutput2, medianKernel, medianOutput3);
    cv::namedWindow("Median filter medianOutput3");
    cv::imshow("Median filter medianOutput3",medianOutput3);
    cv::waitKey(0);
    return 0;
}


void linearFilter(cv::Mat src, cv::Mat k, cv::Mat &output){
    int u, v, i, j, tmp=0;
    cv::Mat padded = cv::Mat::zeros(src.rows+k.rows-1, src.cols+k.cols-1, src.type());
    output.create(src.rows, src.cols, src.type());
    // zero padding on src
    for(u=0; u<src.rows; u++){
        for(v=0; v<src.cols; v++){
            padded.at<uchar>(u+1, v+1) = src.at<uchar>(u, v); 
        }
    }

    for(u=0; u<src.rows; u++){
        for(v=0; v<src.cols; v++){
            for(i=-1; i<=1; i++){
                for(j=-1; j<=1; j++){
                    tmp += padded.at<uchar>(u-i+1, v-j+1) * k.at<float>(i+1, j+1);
                }
            } 
            output.at<uchar>(u, v) = tmp;
            tmp = 0;
        }
    }
 }


void medianFilter(cv::Mat src, cv::Mat k, cv::Mat &output){
    int u, v, i, j, tmp=0;
    std::vector<u_char> adjacentVal;
    cv::Mat padded = cv::Mat::zeros(src.rows+k.rows-1, src.cols+k.cols-1, src.type());
    output.create(src.rows, src.cols, src.type());
    // zero padding on src
    for(u=0; u<src.rows; u++){
        for(v=0; v<src.cols; v++){
            padded.at<uchar>(u+k.rows/2, v+k.cols/2) = src.at<uchar>(u, v); 
        }
    }

    for(u=0; u<src.rows; u++){
        for(v=0; v<src.cols; v++){
            for(i=-k.rows/2; i<=k.rows/2; i++){
                for(j=-k.cols/2; j<=k.cols/2; j++){
                    tmp = padded.at<uchar>(u-i+k.rows/2, v-j+k.cols/2) * k.at<float>(i+k.rows/2, j+k.cols/2);
                    //fprintf(stdout, "u: %d\tv: %d\ti: %d\tj: %d\n", u, v, i, j);
                    if(tmp > 0){adjacentVal.push_back(tmp);};
                    tmp = 0;
                }
            } 
            sort(adjacentVal.begin(), adjacentVal.end());
            output.at<uchar>(u, v) = adjacentVal[adjacentVal.size()/2];
            adjacentVal.clear();
        }
    }   
}