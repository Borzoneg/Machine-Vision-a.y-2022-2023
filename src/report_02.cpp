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

    // applying kernel
    for(u=0; u<src.rows; u++){
        for(v=0; v<src.cols; v++){
            for(i=-k.rows/2; i<=k.rows/2; i++){
                for(j=-k.cols/2; j<=k.cols/2; j++){
                    tmp += padded.at<uchar>(u-i+k.rows/2, v-j+k.cols/2) * k.at<float>(i+k.rows/2, j+k.cols/2);
                }
            } 
            output.at<uchar>(u, v) = tmp;
            tmp = 0;
        }
    }
 }