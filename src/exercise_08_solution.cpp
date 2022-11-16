#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>

using namespace std;
using namespace cv;

/**
 * @brief class2colour Function for converting a class integer to a color
 * @param classInt The class integer. Each number is its own class so the digit "1" also have the class label 1
 * @return Color expressed as a 8U cv::Scalar, which is unique for each class.
 */
Scalar class2colour(int classInt){
	Scalar s = Scalar(0,0,0);
	switch(classInt){
    case 1: s=Scalar(255,255,255);break; //white
	case 2: s=Scalar(255,255,0);break; //cyan
	case 3: s=Scalar(255,0,255);break; //magenta
	case 4: s=Scalar(255,0,0);break; //blue
	case 5: s=Scalar(0,255,255);break; //yellow
	case 6: s=Scalar(0,255,0);break; //green
	case 7: s=Scalar(0,0,255);break; //red
    case 8: s=Scalar(150,150,150);break; //gray
    case 9: s=Scalar(150,151,255);break; //orange
	}
	return s;
}

int main(int argc, char** argv){
    string pathName = "../src/images/numbers/";
    Mat src;
    vector<vector<Point> > contours; //A single contour is a vector<Point>. This is a vector of contours
    vector<Vec4i > hierarchy; //Not used in this example
    vector<vector<vector<Point> > > contoursVector; //vector of 'contours vectors'
    int attributesPrSample = 2; //In this example you will compute perimeter and the mu12 moment, so 2 attributes
    int numberOfClasses = 9; //9 different numbers to ditinguish between
    int numberOfTrainingSamples = 0; //total number of training samples (will be increased as the training images are processed

    /* Load training images (one for each number), and exact the contours */
	for(int i = 0; i<numberOfClasses; i++){
        src = imread(pathName + std::to_string(i+1) +".png",CV_8UC1); //load 3-channel colour image
        findContours(src, contours, hierarchy, cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
		contoursVector.push_back(contours);
		numberOfTrainingSamples += contours.size();
	}


    /* Initialize the training sample and training class matrices */
    Mat trainingSamples(numberOfTrainingSamples, attributesPrSample, CV_32FC1); //each row is a sample, each column an artribute of that sample
    Mat trainingGTClasses(numberOfTrainingSamples,1,CV_32S); //the i'th row is the classification of the i'th sample

    /* Compute perimeter and mu12 moment for each single contour (each contour is a single digit in the training samples).
       Append all attributes and corresponding ground truth class in respectively the trainingSamples matrix and trainingGTClasses matrix */
	double perimeter;
	Moments mu;
	double mu12;
	int index=0;
	for(int c = 0; c<numberOfClasses; c++){
		contours = contoursVector[c];
		for(int i = 0; i<contours.size(); i++){
			perimeter = arcLength(contours[i],true); //calculates the perimeter
			mu = moments(contours[i],false); //calculates all the moments
			mu12 = mu.mu12; //extracts the central moment i=1, j=2
			trainingSamples.at<float>(index,0) = perimeter;
			trainingSamples.at<float>(index,1) = mu12;
            trainingGTClasses.at<int>(index,0) = c+1;
			index++;
		}
	}

    /* Initialize the NormalBayesClass classifier and train the classifier using the trainingSamples and trainingGTClasses */
    Ptr<cv::ml::NormalBayesClassifier> bayesClassifier = cv::ml::NormalBayesClassifier::create();
    bayesClassifier->train(trainingSamples, cv::ml::ROW_SAMPLE, trainingGTClasses);

    /* Load test sudoku and extract contours */
    Mat testSudoku;
    testSudoku = imread("../src/images/testSudoku.png",CV_8UC1);
    findContours(testSudoku, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);


    /* Compute attributes for the test contours and save them in predictionSamples */
	Mat predictionSamples(contours.size(),2,CV_32FC1);
    Mat results(contours.size(),1,CV_32S);
	for(int i = 0; i<contours.size(); i++){
		perimeter = arcLength(contours[i],true); //calculates the perimeter
		mu = moments(contours[i],false); //calculates all the moments
		mu12 = mu.mu12; //extracts the central moment i=1, j=2
		predictionSamples.at<float>(i,0) = perimeter;
		predictionSamples.at<float>(i,1) = mu12;
	}

    /* Use the trained OpenCV classifier to predict the class of each test sample */
    bayesClassifier->predict(predictionSamples, results);

    /* Draw each test contour with a color determined by the class predicted by the classifier */
    Mat contourImg = Mat::zeros(testSudoku.size(), CV_8UC3);
	int classInt;
	for(int i = 0; i<contours.size();i++){
        classInt = (int)(results.at<int>(i,0));
		drawContours(contourImg,contours, i, class2colour(classInt));
	}

    /* Show the original sudoku and the colored contours */
	namedWindow("TestSudoku");
	imshow("TestSudoku", testSudoku);
	namedWindow("Contour");
	imshow("Contour", contourImg);
	waitKey(0);

	return 1;
}
