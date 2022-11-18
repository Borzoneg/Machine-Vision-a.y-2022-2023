#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }
    Mat image, img2, img3;
    image = imread( argv[1], 1 );
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    // namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image);

    //cout<<image.type()<<endl;
    
    image.convertTo(img2, -1, 1, 50); // something, contrast, brightness
    // imshow("Changed brighness", img2);

    Mat matRotation = getRotationMatrix2D(Point(image.rows/2, image.cols/2), 180, 1);
    warpAffine(image, img3, matRotation, image.size());
    //imshow("Changed rotation", img3);

    // Exercise 1.3 image.jpg needs to be rotated 90 degrees. Implement a rotation algorithm using single pixel access.
    Mat rotatedPxbyPx(image.cols, image.rows, image.type());
    int i, j;
    for(i=0; i<image.rows; i++){
        for(j=0; j<image.cols; j++){
            rotatedPxbyPx.at<Vec3b>(j, image.rows-i+1) = image.at<Vec3b>(i, j);
        }
    }
    // imshow("Pixel by pixel", rotatedPxbyPx);

    // Two ways to access pixels
    // cv::Vec3b* data = image.ptr<Vec3b>(i);
    // data[j][0] = 0; //set blue to 0

    Mat src_hls;
    cvtColor(image, src_hls, COLOR_BGR2HLS);

    vector<Mat> bgrChannels, hlsChannels;
    split(image,bgrChannels);
    split(src_hls,hlsChannels);

    int flags = WINDOW_AUTOSIZE;
    namedWindow("b",flags);
    namedWindow("g",flags);
    namedWindow("r",flags);
    namedWindow("h",flags);
    namedWindow("l",flags);
    namedWindow("s",flags);

    imshow("b",bgrChannels[0]);
    imshow("g",bgrChannels[1]);
    imshow("r",bgrChannels[2]);
    imshow("h",hlsChannels[0]);
    imshow("l",hlsChannels[1]);
    imshow("s",hlsChannels[2]);
    
    // Exercise 1.4 Try to segment the red spoons. First: convert to a suitable color
    // space. Next: apply a threshold. Display the segmented spoon(s)
    for(i=0; i<image.rows; i++){
        for(j=0; j<image.cols; j++){
            if(image.at<Vec3b>(i,j)[2] < 100){
                image.at<Vec3b>(i,j)[0] = 0; //set blue to 0
                image.at<Vec3b>(i,j)[1] = 0; //set blue to 0
                image.at<Vec3b>(i,j)[2] = 0; //set blue to 0
            }
            if(image.at<Vec3b>(i,j)[0] > 90){
                image.at<Vec3b>(i,j)[0] = 0; //set blue to 0
                image.at<Vec3b>(i,j)[1] = 0; //set blue to 0
                image.at<Vec3b>(i,j)[2] = 0; //set blue to 0
            }
            if(image.at<Vec3b>(i,j)[1] > 60){
                image.at<Vec3b>(i,j)[0] = 0; //set blue to 0
                image.at<Vec3b>(i,j)[1] = 0; //set blue to 0
                image.at<Vec3b>(i,j)[2] = 0; //set blue to 0
            }
        }
    }
    imshow("Display Image", image);

    waitKey(0);
    return 0;
}