#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <numeric>

double computeVariation(std::vector<uchar> data);
cv::Mat paddingImg(cv::Mat img, int paddingSize);
float computeOtsuCriteria(cv::Mat img, int treshold);
cv::Mat erodeImg(cv:: Mat img, cv::Mat kernel);
cv::Mat findConnectedComponents(cv:: Mat img);

int main(int argc, char* argv[]){
    cv::Mat img = cv::imread("../imgs/licencePlate1.png", cv::IMREAD_GRAYSCALE);
/* 
    // there is a bug in the otsu method or in the erosion implemantion

    //====================================OTSU Method===================================//
    cv::Mat customOtsuImg = img.clone(), otsuImg = img.clone();
    double treshold, bestTreshold = 256;
    float otsuCriteria, bestOtsuCriteria = FLT_MAX;

    for(treshold=0; treshold<256; treshold++){
         otsuCriteria = computeOtsuCriteria(img, treshold);
         if(otsuCriteria < bestOtsuCriteria){
             bestOtsuCriteria = otsuCriteria;
             bestTreshold = treshold;
         }
    }
    std::cout<<"Best Otsu criteria: "<<bestOtsuCriteria<<" selected treshold: "<<bestTreshold<<std::endl;
    
    cv::threshold(img, customOtsuImg, bestTreshold, 255, 0);
    cv::threshold(img, otsuImg, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    // cv::imshow("Treshold custom Otsu", customOtsuImg);
    // cv::imshow("Treshold Otsu", otsuImg);
    
    
    //==============================EROSION IMPLEMENTATION==============================//
    cv::Mat kernel = (cv::Mat_<uchar>(3,3) << 0, 1, 0, 1, 1, 1, 0, 1, 0);
    
    cv::Mat example = (cv::Mat_<uchar>(5,5) << 0, 0, 0, 0, 0,
                                               0, 1, 1, 1, 0, 
                                               0, 1, 1, 1, 1,
                                               0, 1, 1, 1, 0,
                                               0, 1, 1, 0, 0);
    cv::Mat erodedExample = example.clone();
    cv::erode(example, erodedExample, kernel);
    cv::Mat customeErodedExample = erodeImg(example, kernel);

    cv::Mat customErodedImg = erodeImg(customOtsuImg, kernel), erodedImg = img.clone();
    cv::erode(otsuImg, erodedImg, kernel);
    cv::imshow("Original tresholded image", otsuImg);
    cv::imshow("Eroded custom", customErodedImg);
    cv::imshow("Eroded", erodedImg);  */
 
    //==============================CONNECTED COMPONENTS==============================//
    // it doesn't works on images because is overflowing the memory allocated for the stack(?)
    
    cv::Mat exampleCc = (cv::Mat_<uchar>(8,14) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                               0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0,
                                               0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0,
                                               0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
                                               0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                                               0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                                               0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
                                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                                               );

    std::cout<<exampleCc<<"\n\n";
    cv::Mat customConnectedCompontenImg = findConnectedComponents(exampleCc);
    std::cout<<customConnectedCompontenImg<<std::endl;
    return 0;
}


float computeOtsuCriteria(cv::Mat img, int treshold){
    cv::Mat tresholdedImg(img.size(), img.type());
    int i, j;
    float weight0, weight1, variance0 , variance1;
    float nPixels = img.rows * img.cols, nonZeroPixels = 0;
    std::vector <uchar> pixelsAbove;
    std::vector <uchar> pixelsBelow;


    for(i=0; i<img.rows; i++){ // setup the tresholded image
        for(j=0; j<img.cols; j++){
            if(img.at<uchar>(i, j) >= treshold){
                tresholdedImg.at<uchar>(i, j) = 1;
                nonZeroPixels ++; // count how many pixel are above treshold
            }
            else
                tresholdedImg.at<uchar>(i, j) = 0;
        }
    }

    weight1 = nonZeroPixels / nPixels;
    weight0 = 1- weight1;
    if(weight0 == 0 || weight1 == 0)
        return FLT_MAX; // if we have one of the weigth to zero it means that all pixels are above or below, so we discard this treshold
    
    for(i=0; i<img.rows; i++){
        for(j=0; j<img.cols; j++){
            if(tresholdedImg.at<uchar>(i, j) == 1)
                pixelsAbove.push_back(img.at<uchar>(i,j));
            else
                pixelsBelow.push_back(img.at<uchar>(i,j));
        }
    }
    
    variance0 = pixelsBelow.size() > 0 ? computeVariation(pixelsBelow) : 0;
    variance1 = pixelsAbove.size() > 0 ? computeVariation(pixelsAbove) : 0;

    //std::cout<<"Non zero pixels: "<<nonZeroPixels<<" Number of pixels: "<<nPixels<<" Weight0: "<<weight0<<" Weight1: "<<weight1<<" Variance0: "<<variance0<<" Variance1: "<<variance1<<std::endl;
    return weight0*variance0 + weight1*variance1; 
}


cv::Mat erodeImg(cv:: Mat img, cv::Mat kernel){
    int i, j, k, l, sum = 0;
    // the size of the kernel divided by 2 will be our correction factor to scale the padded image and to offset by the right amount when applying the kernel
    int corr = kernel.rows/2;
    cv::Mat eroded (img.rows, img.cols, img.type());
    cv::Mat paddedImg = paddingImg(img, corr);

    for(i=corr; i<paddedImg.rows-corr; i++){ // for cycle to apply a kernel to a image
         for(j=corr; j<paddedImg.cols-corr; j++){
            for(k=-corr; k<=corr; k++){
                for(l=-corr; l<=corr; l++){
                    sum += (kernel.at<uchar>(k+corr, l+corr) * paddedImg.at<uchar>(i+k, j+l));   
                }
            }
            // TODO : space to generalize 5 and 255 to sum of kernel
            eroded.at<uchar>(i-corr, j-corr) = sum >= 5*255 ? 255 : 0; // if sum is 5 then all the neighbor of the pixel are at one
            sum = 0;     
        }
    }
    return eroded;
}


cv::Mat findConnectedComponents(cv:: Mat img){ // using linked list should be better
    int i, j, k , l, label = 2;
    bool propagated = false;
    std:: vector <std::vector<cv::Point>>pixelByLabel;
    cv::Mat cCImg = img.clone();
    
    for(i=0; i<cCImg.rows; i++){ 
        for(j=0; j<cCImg.cols; j++){
            if(cCImg.at<uchar>(i, j) != 0){ // we want to apply the algorithm only to non zero pixel
                for(k=0; k<4;k++){ // if one of the pixel before the one we re taking into account is different from zero we will inherit the first we can find
                    if(i-1+k/3 < 0) // check if we re not out of bond
                        break;
                    if(j-1+k%3 < 0) // check if we re not out of bond
                        break;
                    if(cCImg.at<uchar>(i-1+k/3, j-1+k%3) != 0){ // if we find some component before the pixel
                        cCImg.at<uchar>(i, j) = cCImg.at<uchar>(i-1+k/3, j-1+k%3); // propagate the label
                        pixelByLabel[cCImg.at<uchar>(i-1+k/3, j-1+k%3)-2].push_back(cv::Point(i,j)); // we add to the vector associated to the label for future use
                        propagated = true;
                        break; // the first label is fine
                    }                    
                }                
                if(!propagated){ // if we haven't propagated we have all zero neighbor
                    std::vector <cv::Point>newLabelVec; // we create a new vector for this label and push it to the general vector
                    newLabelVec.push_back(cv::Point(i, j));
                    pixelByLabel.push_back(newLabelVec);
                    
                    cCImg.at<uchar>(i, j) = label++;    
                }
                propagated = false;
            }
        }
    }
    // second loop to merge labels    
    std::vector <cv::Point>tmpVec;
    for(i=0; i<cCImg.rows; i++){ 
        for(j=0; j<cCImg.cols; j++){
            if(cCImg.at<uchar>(i, j) != 0){ 
                for(k=0; k<4;k++){
                    if(i-1+k/3 < 0) // check if we re not out of bond
                        break;
                    if(j-1+k%3 < 0) // check if we re not out of bond
                        break;
                    if(cCImg.at<uchar>(i-1+k/3, j-1+k%3) != cCImg.at<uchar>(i, j)){ // if one of the neighbor has a different label it will inherit our label
                        tmpVec = pixelByLabel[cCImg.at<uchar>(i-1+k/3, j-1+k%3) - 2]; // we retrieve the correct vector
                        for(l=0; l < tmpVec.size(); l++){ // let's change all the point in the vector to the new label and add them to the vector associated to it
                            cCImg.at<uchar>(tmpVec[l].x, tmpVec[l].y) = cCImg.at<uchar>(i, j);
                            pixelByLabel[cCImg.at<uchar>(i, j)-2].push_back(tmpVec[l]); 
                        }
                    }                 
                }
            }
            
        }
    }
    return cCImg;
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



double computeVariation(std::vector<uchar> data){
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    double mean = sum / data.size();
    double sq_sum = std::inner_product(data.begin(), data.end(), data.begin(), 0.0);
    double stdev = sq_sum / data.size() - mean * mean;
    return stdev;
}