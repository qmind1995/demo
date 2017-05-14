//
// Created by tri on 02/05/2017.
//

#include <stdio.h>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <flann/flann.hpp>
#include <opencv/cv.hpp>

#include "bundleAdjustment.cpp"
//#include "nViewStructureFromMotion.cpp"

using namespace std;
using namespace cv;

Mat normalization_fundamental(vector<Point2d> points, vector<Point2d>& points_nor){
    unsigned long len = points.size();
    float x_centroid = 0;
    float y_centroid = 0;
    for(int i=0; i<len; i++){
        x_centroid += points[i].x;
        y_centroid += points[i].y;
    }
    x_centroid = x_centroid/len;
    y_centroid = y_centroid/len;

    for(int i=0; i<len; i++){
        points_nor[i].x = points[i].x - x_centroid;
        points_nor[i].y = points[i].y - y_centroid;
    }

    float average_distant_from_origin = 0;
    for(int i=0; i<len; i++){
        average_distant_from_origin += sqrt(points_nor[i].x*points_nor[i].x + points_nor[i].y*points_nor[i].y);
    }
    average_distant_from_origin = average_distant_from_origin/len;

    for(int i=0; i<len; i++){
        points_nor[i].x = points_nor[i].x / (average_distant_from_origin / sqrt(2.0));
        points_nor[i].y = points_nor[i].y / (average_distant_from_origin / sqrt(2.0));
    }

    Mat res(3, 3, CV_64F);
    res.at<double>(0, 0) = sqrt(2) / average_distant_from_origin;
    res.at<double>(0, 1) = 0;
    res.at<double>(0, 2) = - x_centroid * sqrt(2) / average_distant_from_origin;

    res.at<double>(1, 0) = 0;
    res.at<double>(1, 1) = sqrt(2) / average_distant_from_origin;
    res.at<double>(1, 2) = - y_centroid * sqrt(2) / average_distant_from_origin;

    res.at<double>(2, 0) = 0;
    res.at<double>(2, 1) = 0;
    res.at<double>(2, 2) = 1;

    return  res;
}

/**
 *
 * @param imgsPath
 * @param firstR
 * @param firstT
 * @param Ks
 * @param showMatching
 * @param ceilViewRange
 * @return
 */
vector<Point3d> get3DPoints(vector<string> imgsPath, Mat &firstR, Mat &firstT , Mat K, bool showMatching, int ceilViewRange = 100){

    int NView = imgsPath.size();
    // Matching


    //get camera pose



    vector<Point3d> points3D;

//    BAForMultiViews()
    return points3D;
}