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

vector< cv::Point3d > get3DPointsFromTwoImg(string image_1_address, string image_2_address, Mat K,bool showMatching, int ceilViewRange =100){

    Mat img_1 = imread( image_1_address, IMREAD_GRAYSCALE );
    Mat img_2 = imread( image_2_address, IMREAD_GRAYSCALE );

    Ptr<BRISK> detector = BRISK::create();

    vector<KeyPoint> keypoints_0, keypoints_1;
    Mat descriptors_1, descriptors_2;
    detector->detectAndCompute( img_1, Mat(), keypoints_0, descriptors_1 );
    detector->detectAndCompute( img_2, Mat(), keypoints_1, descriptors_2 );

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::makePtr<cv::FlannBasedMatcher>(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
    vector< DMatch > matchesOneToTwo;
    vector< DMatch > matchesTwoToOne;
    vector< DMatch > matches;
    matcher->match( descriptors_1, descriptors_2, matchesOneToTwo, Mat() );
    //matcher->match( descriptors_2, descriptors_1, matchesTwoToOne, Mat() );

    //for(int i = 0; i < descriptors_1.rows; i++){
    //    if(matchesTwoToOne[matchesOneToTwo[i].trainIdx].trainIdx == i){
    //        matches.push_back(DMatch(i, matchesOneToTwo[i].trainIdx, matchesOneToTwo[i].distance));
    //    }
    //}
    matches = matchesOneToTwo;
    sort(matches.begin(), matches.end());

    int view_range = min((int)matches.size(), ceilViewRange);
    std::vector< DMatch > good_matches;
    for(int i=0; i<view_range; i++)
        good_matches.push_back(matches[i]);

    unsigned long point_count = (unsigned long)view_range;
    vector<Point2d> points0(point_count);
    vector<Point2d> points1(point_count);

    for(int i=0; i<point_count; i++){
        float x1 = keypoints_0[good_matches[i].queryIdx].pt.x;
        float y1 = keypoints_0[good_matches[i].queryIdx].pt.y;

        float x2 = keypoints_1[good_matches[i].trainIdx].pt.x;
        float y2 = keypoints_1[good_matches[i].trainIdx].pt.y;

        points0[i] = Point2d(x1, y1);
        points1[i] = Point2d(x2, y2);
    }

    vector<Point2d> points0_nor(point_count);
    vector<Point2d> points1_nor(point_count);

    Mat T1 = normalization_fundamental(points0, points0_nor);
    Mat T2 = normalization_fundamental(points1, points1_nor);

    Mat inliner;
    double dis_outline = 2*T1.at<double>(0,0);

    Mat fundamental_mat = findFundamentalMat(points0_nor, points1_nor, FM_RANSAC, dis_outline, 0.8, inliner);

    fundamental_mat = T1.t()*fundamental_mat*T2;
    Mat essential_mat = K.t()*fundamental_mat*K;

    vector<Point2d> points0_good;
    vector<Point2d> points1_good;

    for(int i=0; i<point_count; i++){
        if ((unsigned int)inliner.at<char>(i) == 1){
            points0_good.push_back(points0[i]);
            points1_good.push_back(points1[i]);
        }
    }
    
    Mat rotation, translation;
    recoverPose(essential_mat, points0_good, points1_good, K, rotation, translation);

    if(showMatching){
        Mat img_matches;
        drawMatches( img_1, keypoints_0,
                 img_2, keypoints_1,
                 good_matches,
                 img_matches, Scalar::all(-1), Scalar::all(-1),
                 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        imshow( "Good Matches", img_matches );
        imwrite("res.png", img_matches);

    }

    vector< cv::Point3d > points3D = bundleAdjustmentForTwoViews(points0_good, points1_good, rotation, translation, K, view_range);

    return points3D;
}
