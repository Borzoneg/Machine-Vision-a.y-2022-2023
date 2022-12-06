#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>


int main(int argc, char **argv) {
    cv::Mat boxInScene = cv::imread("../inputFiles/imgs/box_in_scene.png", cv::IMREAD_GRAYSCALE);
    cv::Mat box = cv::imread("../inputFiles/imgs/box.png", cv::IMREAD_GRAYSCALE);

    cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();

    std::vector<cv::KeyPoint> keyPointsScene;
    std::vector<cv::KeyPoint> keyPointsBox;
    cv::Mat descriptorsScene, descriptorsBox;

    detector->detect(boxInScene, keyPointsScene); // detect the keypoint of the image and put them in keypoint
    detector->compute(boxInScene, keyPointsScene, descriptorsScene); // detect the descriptor (based on colors, shape, gradient, ...)
    detector->detectAndCompute(box, cv::noArray(), keyPointsBox, descriptorsBox); // both operation together

    cv::Mat imgSceneKeypoints, imgBoxKeyPoints;
    cv::drawKeypoints(boxInScene, keyPointsScene, imgSceneKeypoints); // draw the keypoins in both images and show'em
    cv::drawKeypoints(box, keyPointsBox, imgBoxKeyPoints);
    cv::imshow("Keypoints detected in box in scene", imgSceneKeypoints);
    cv::imshow("Keypoints detected in box alone", imgBoxKeyPoints);
    

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE); // create a bruteforce-based matcher
    std::vector<cv::DMatch> matches;
    matcher->match(descriptorsScene, descriptorsBox, matches); // try to match the 2 set of keypoints based on descriptors

    cv::Mat imgMatches; // draw the matches and show'em
    cv::drawMatches(boxInScene, keyPointsScene, box, keyPointsBox, matches, imgMatches);
    cv::imshow("Matches between the two images", imgMatches);
    

    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher->knnMatch(descriptorsScene, descriptorsBox, knnMatches, 2); 
    float ratio_tresh = 0.7;
    std::vector<cv::DMatch> filteredMatches;
    for(auto matches : knnMatches){
        if(matches[0].distance < ratio_tresh*matches[1].distance){
            filteredMatches.push_back(matches[0]);
        }
    }
    cv::Mat imgFilteredMatches; // draw the good matches and show'em
    cv::drawMatches(boxInScene, keyPointsScene, box, keyPointsBox, filteredMatches, imgFilteredMatches);
    cv::imshow("Filtered matches between the two images", imgFilteredMatches);
    

    // create the vector for the points that had matches in the two imgs
    std::vector<cv::Point2f> pointsBox;
    std::vector<cv::Point2f> pointsScene;
    
    for(auto match:filteredMatches){ // populate the two vectors
        pointsScene.push_back(keyPointsScene[match.queryIdx].pt); // query train refers to the matches and are based on the order we inserted
        pointsBox.push_back(keyPointsBox[match.trainIdx].pt); // the descriptors in the matcher call query->scene; train->box
    }
    cv::Mat H = cv::findHomography(pointsBox, pointsScene, cv::RANSAC); // find the homography knowing that pointBox[i] is the same point as pointsScene[i]
    
    // the corners for the box image are the four corner of the image
    std::vector<cv::Point2f> cornersBox(4), cornersScene(4);
    cornersBox[0] = cv::Point2f(0, 0);
    cornersBox[1] = cv::Point2f((float)box.cols, 0);
    cornersBox[2] = cv::Point2f((float)box.cols, (float)box.rows);
    cornersBox[3] = cv::Point2f(0, (float)box.rows);
    
    cv::perspectiveTransform(cornersBox, cornersScene, H); // apply the homography to those corners to retrieve them in the scene 

    // draw a rectangle to show the found box
    cv::Mat detectedBox = boxInScene.clone();
    cv::line(detectedBox, cornersScene[0], cornersScene[1], cv::Scalar(0, 255, 0), 2);
    cv::line(detectedBox, cornersScene[1], cornersScene[2], cv::Scalar(0, 255, 0), 2);
    cv::line(detectedBox, cornersScene[2], cornersScene[3], cv::Scalar(0, 255, 0), 2);
    cv::line(detectedBox, cornersScene[3], cornersScene[0], cv::Scalar(0, 255, 0), 2);
    
    cv::imshow("Detected object", detectedBox );
    cv::waitKey();
    return 0;
}
