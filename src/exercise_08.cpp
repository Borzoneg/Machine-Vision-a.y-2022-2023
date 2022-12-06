#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <fstream>


cv::Scalar class2colour(int classInt);


int main(int argc, char** argv){
    int i, j, k;
    cv::Mat img;
    double perimeter, area, totalRowsData = 0, nFeatures = 2;
    cv::Moments moments;
    std::vector<std::vector<cv::Point>> contours; 
    std::vector<double> perimeters;
    std::vector<double> areas;
    std::vector<cv::Moments> momentsNumber;
    std::vector<std::vector<cv::Moments>> momentsNumbers;
    std::vector<cv::Mat> imgNumbers;
    std::vector<std::vector<std::vector<cv::Point>>> contoursNumbers;
    std::vector<std::vector<double>> perimetersNumbers;
    std::vector<std::vector<double>> areasNumbers;
    std::vector<cv::Vec4i > hierarchy; //Not used in this example

    for(i=1; i<10; i++){
        img = cv::imread("../inputFiles/imgs/numbers/" + std::to_string(i) + ".png", CV_8UC1);
        cv::findContours(img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE); // find every contours in each img
        for(auto contour : contours){ // for each contour found
            perimeter = cv::arcLength(contour, true); // find perimeter
            area  = cv::contourArea(contour); // find area
            moments = cv::moments(contour);
            perimeters.push_back(perimeter);  // add them in their vectors
            areas.push_back(area); // add them in their vectors
            momentsNumber.push_back(moments);
        }
        // add each vector of "feature" in the proper database
        totalRowsData += contours.size();
        imgNumbers.push_back(img);
        contoursNumbers.push_back(contours);        
        perimetersNumbers.push_back(perimeters);    
        areasNumbers.push_back(areas);              
        momentsNumbers.push_back(momentsNumber);

        // clean the intermediate vectors
        contours.clear();
        perimeters.clear();
        areas.clear();
        momentsNumber.clear();
    }
    // at this point i have each vector of size 9, with perimetersNumbers[1] being all the perimeters founded for digit 2
    // the number of perimeters stored in perimetersNumber[1] can differ from the number of perimeters stored in perimetersNumber[2]
    // but the number of areas in areasNumbers[0] is equal to the number of perimeters in perimetersNumber[1]
    std::cout<<"\tcontours\tperimeters\tareas\tmoments\n";
    for(int i=0; i<9;i++){ // printing size of each parameters for each number
        std::cout<< i+1 <<": \t"<<contoursNumbers[i].size() << "\t\t" << perimetersNumbers[i].size() << "\t\t" << areasNumbers[i].size() << " \t" << momentsNumbers[i].size() << "\n";
    }
    // create the trainData in an acceptable format and labels as well
    cv::Mat trainData(totalRowsData, nFeatures, CV_32FC1);
    cv::Mat labels(totalRowsData, 1, CV_32S);
    k = 0;
    std::ofstream Filedata("../outputFiles/datas.txt");
    Filedata<<"\tperimeter\tmoment\n";
    for(i=0; i<9; i++){
        for(j=0; j<perimetersNumbers[i].size(); j++, k++){
            trainData.at<float>(k, 0) = perimetersNumbers[i][j];
            trainData.at<float>(k, 1) = momentsNumbers[i][j].mu12;
            labels.at<int>(k) = i+1;
            Filedata<<labels.at<int>(k)<<":\t"<<trainData.at<float>(k, 0)<< "\t\t" <<trainData.at<float>(k, 1)<<"\n";
        }
    }
    cv::Ptr<cv::ml::NormalBayesClassifier> bayesClassifier = cv::ml::NormalBayesClassifier::create(); // create the classifier
    bayesClassifier->train(trainData, cv::ml::ROW_SAMPLE, labels); // and train it

    // repeat the previous step on the test data, extract the feature and construct a dataset with them
     cv::Mat imgTest = cv::imread("../inputFiles/imgs/testSudoku.png", CV_8UC1);
    // cv::Mat imgTest = cv::imread("../img/handwroteNumbers.png", CV_8UC1); // the image hsa to "cleaner" to get some result
    cv::findContours(imgTest, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Mat testData(contours.size(), nFeatures, CV_32FC1);
    k = 0;
    for(auto contour : contours){
        testData.at<float>(k, 0) = cv::arcLength(contour, true); 
        testData.at<float>(k, 1) = cv::moments(contour).mu12;
        k++;
    }
    cv::Mat results(contours.size(),1,CV_32S);
    bayesClassifier->predict(testData, results); // predict and put the labels in results
    
    cv::Mat imgColoredNumbers = cv::Mat::zeros(imgTest.size(), CV_8UC3);
	int classInt;
	for(int i = 0; i<contours.size();i++){ // draw the contours with the color associated with its own class aka number
        classInt = (int)(results.at<int>(i,0));
		drawContours(imgColoredNumbers,contours, i, class2colour(classInt));
	}

    cv::imshow("Sudoku", imgTest);
    cv::imshow("Sudoku digit detected", imgColoredNumbers);
    cv::waitKey(0);
    return 0;
}

cv::Scalar class2colour(int classInt){
	cv::Scalar s = cv::Scalar(0,0,0);
	switch(classInt){
    case 1: s=cv::Scalar(255,255,255);break; //white
	case 2: s=cv::Scalar(255,255,0);break; //cyan
	case 3: s=cv::Scalar(255,0,255);break; //magenta
	case 4: s=cv::Scalar(255,0,0);break; //blue
	case 5: s=cv::Scalar(0,255,255);break; //yellow
	case 6: s=cv::Scalar(0,255,0);break; //green
	case 7: s=cv::Scalar(0,0,255);break; //red
    case 8: s=cv::Scalar(150,150,150);break; //gray
    case 9: s=cv::Scalar(150,151,255);break; //orange
	}
	return s;
}