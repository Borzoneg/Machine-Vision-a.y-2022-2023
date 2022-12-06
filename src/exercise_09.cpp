#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>

int main(int argc, char *argv[]){
    cv::Mat img = cv::imread("../imgs/obama.png", cv::IMREAD_GRAYSCALE);
    std::vector<cv::Rect> faces;
    cv::equalizeHist(img, img); // normalization of image
    cv::CascadeClassifier face_classifier, eyes_classifier;
    if(!face_classifier.load("../inputFiles/haarcascades/haarcascade_frontalface_alt.xml")) return 1;
    if(!eyes_classifier.load("../inputFiles/haarcascades/haarcascade_eye_tree_eyeglasses.xml")) return 1;
    face_classifier.detectMultiScale(img, faces, 1.1, 2, 0|cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
    for (int i = 0; i < faces.size(); i++){ // for each face detected
        cv::Point center(faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2); // find the center
        cv::ellipse(img, center, cv::Size(faces[i].width/2, faces[i].height/2), 0, 0, 360, cv::Scalar(255, 0, 255), 4, 8, 0); // draw an ellipse around it
        cv::Mat faceROI = img(faces[i]);
        std::vector<cv::Rect> eyes; // create a vector for the eyes
        eyes_classifier.detectMultiScale(faceROI, eyes, 1.1, 2, 0 |cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
        for (size_t j = 0; j < eyes.size(); j++){ // for each eyes detected
            cv::Point eye_center(faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2); // find enter
            int radius = cvRound((eyes[j].width + eyes[j].height)*0.25); 
            circle(img, eye_center, radius, cv::Scalar(255, 0, 0), 4, 8, 0); // draw a circle around them
        }
    }
    imshow("Detected face", img);
    cv::waitKey();
    return 0;
}