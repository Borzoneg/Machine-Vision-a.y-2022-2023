#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main(int argc, char** argv )
{
    int width1 = 1024, height1 = 768, nCh1 = 1, nBitperCh1 = 8;
    int size1 = width1 * height1 * nCh1 * nBitperCh1;
    cout<<"Size image1: "<<size1<<" bit = "<<size1/8<<" Byte"<<endl;
    Mat image1(1024, 768, CV_8UC1);

    int width2 = 640, height2 = 480, nCh2 = 3, nBitperCh2 = 32;
    int size2 = width2 * height2 * nCh2 * nBitperCh2;
    cout<<"Size image2: "<<size2<<" bit = "<<size2/8<<" Byte"<<endl;
    Mat image2(640, 480, CV_32FC3);

    int width3 = 1280, height3 = 720, nCh3 = 2, nBitperCh3 = 16;
    int size3 = width3 * height3 * nCh3 * nBitperCh3;
    cout<<"Size image3: "<<size3<<" bit = "<<size3/8<<" Byte"<<endl;
    Mat image3(1280, 720, CV_16SC2);

    waitKey(0);
    return 0;
}