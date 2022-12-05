int countNumberOfRice(cv::Mat src){
    cv::Mat resultImage = src.clone();
    cv::Mat kernel = getStructuringElement(cv::MorphShapes::MORPH_CROSS, cv::Size(3, 3), cv::Point(-1,-1));
    int iteration = 1; 
    cv::threshold(src, resultImage, 0, 255, cv::THRESH_BINARY|cv::THRESH_OTSU);
    cv::erode(resultImage, resultImage, kernel, cv::Point(-1,-1), iteration);
    cv::dilate(resultImage, resultImage, kernel, cv::Point(-1,-1), iteration);
    cv::imshow("asd",resultImage);
    cv::waitKey();
    return cv::connectedComponents(resultImage, resultImage);
}
