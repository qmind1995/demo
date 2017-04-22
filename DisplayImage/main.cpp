/*
 * @file SURF_FlannMatcher
 * @brief SURF detector + descriptor + FLANN Matcher
 * @author A. Huaman
 */
#include <stdio.h>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include "opencv2/core.hpp"
#include <cvsba/cvsba.h>
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <flann/flann.hpp>
#include <flann/algorithms/hierarchical_clustering_index.h>
#include <opencv/cv.hpp>

#include "flann/flann.h"
#include "flann/algorithms/lsh_index.h"
#include "bundleAdjustment.cpp"
#include "displayPointCloud.cpp"

using namespace std;
using namespace cv;
void readme();
/*
 * @function main
 * @brief Main function
 */

//return T matrix
Mat normailization_for_fundamental(vector<Point2d>& points){
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
        points[i].x -= x_centroid;
        points[i].y -= y_centroid;
    }

    float average_distant_from_origin = 0;
    for(int i=0; i<len; i++){
        average_distant_from_origin += sqrt(points[i].x*points[i].x + points[i].y*points[i].y);
    }
    average_distant_from_origin = average_distant_from_origin/len;

    for(int i=0; i<len; i++){
        points[i].x = points[i].x / (average_distant_from_origin / sqrt(2.0));
        points[i].y = points[i].y / (average_distant_from_origin / sqrt(2.0));
    }

    Mat res(3, 3, CV_64F);
    res.at<double>(0, 0) = average_distant_from_origin;
    res.at<double>(0, 1) = 0;
    res.at<double>(0, 2) = average_distant_from_origin*(-1)*x_centroid;

    res.at<double>(1, 0) = 0;
    res.at<double>(1, 1) = average_distant_from_origin;
    res.at<double>(1, 2) = average_distant_from_origin*(-1)*y_centroid;

    res.at<double>(2, 0) = 0;
    res.at<double>(2, 1) = 0;
    res.at<double>(2, 2) = 1;

    return  res;
}

cv::Mat rot2euler(const cv::Mat & rotationMatrix) {
    cv::Mat euler(3,1,CV_64F);

    double m00 = rotationMatrix.at<double>(0,0);
    double m02 = rotationMatrix.at<double>(0,2);
    double m10 = rotationMatrix.at<double>(1,0);
    double m11 = rotationMatrix.at<double>(1,1);
    double m12 = rotationMatrix.at<double>(1,2);
    double m20 = rotationMatrix.at<double>(2,0);
    double m22 = rotationMatrix.at<double>(2,2);

    double x, y, z;

    // Assuming the angles are in radians.
    if (m10 > 0.998) { // singularity at north pole
        x = 0;
        y = CV_PI/2;
        z = atan2(m02,m22);
    }
    else if (m10 < -0.998) { // singularity at south pole
        x = 0;
        y = -CV_PI/2;
        z = atan2(m02,m22);
    }
    else
    {
        x = atan2(-m12,m11);
        y = asin(m10);
        z = atan2(-m20,m00);
    }

    euler.at<double>(0) = x;
    euler.at<double>(1) = y;
    euler.at<double>(2) = z;

    return euler;
}

int main( int argc, char** argv )
{
    string image_1_address = "./pictures/dinoRing/dinoR0001.png";//argv[1];
    string image_2_address = "./pictures/dinoRing/dinoR0002.png";//argv[2];
    Mat img_1 = imread( image_1_address, IMREAD_GRAYSCALE );
    Mat img_2 = imread( image_2_address, IMREAD_GRAYSCALE );

//    if( !img_1.data || !img_2.data )
//    { std::cout<< " --(!) Error reading images " << std::endl; return -1; }

    //Detect the keypoints using Detector, compute the descriptors
    Ptr<BRISK> detector = BRISK::create();

    std::vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    detector->detectAndCompute( img_1, Mat(), keypoints_1, descriptors_1 );
    detector->detectAndCompute( img_2, Mat(), keypoints_2, descriptors_2 );


    //Matching descriptor vectors using FLANN matcher and check symmetric
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::makePtr<cv::FlannBasedMatcher>(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
    std::vector< DMatch > matchesOneToTwo;
    std::vector< DMatch > matchesTwoToOne;
    std::vector< DMatch > matches;
    matcher->match( descriptors_1, descriptors_2, matchesOneToTwo, Mat() );
    matcher->match( descriptors_2, descriptors_1, matchesTwoToOne, Mat() );

    for(int i = 0; i < descriptors_1.rows; i++){
        if(matchesTwoToOne[matchesOneToTwo[i].trainIdx].trainIdx == i){
            matches.push_back(DMatch(i, matchesOneToTwo[i].trainIdx, matchesOneToTwo[i].distance));
        }
    }


    //sort match
    sort(matches.begin(), matches.end());

    int view_range = min((int)matches.size(), 100);
    std::vector< DMatch > good_matches;
    for(int i=0; i<view_range; i++)
        good_matches.push_back(matches[i]);

    unsigned long point_count = (unsigned long)view_range;
    vector<Point2d> points1(point_count);
    vector<Point2d> points2(point_count);

    for(int i=0; i<point_count; i++){
        float x1 = keypoints_1[good_matches[i].queryIdx].pt.x;
        float y1 = keypoints_1[good_matches[i].queryIdx].pt.y;

        float x2 = keypoints_2[good_matches[i].trainIdx].pt.x;
        float y2 = keypoints_2[good_matches[i].trainIdx].pt.y;

        points1[i] = Point2d(x1, y1);
        points2[i] = Point2d(x2, y2);
    }

    Mat T1 = normailization_for_fundamental(points1);
    Mat T2 = normailization_for_fundamental(points2);

    Mat fundamental_mat = findFundamentalMat(points1, points2, FM_RANSAC, 3, 0.99);

    fundamental_mat = T1.t()*fundamental_mat*T2;

//    cout << "T1 Matrix = "<< endl << " "  << T1 << endl << endl;
//    cout << "T2 Matrix = "<< endl << " "  << T2 << endl << endl;
//    cout << "Fundamental Matrix = "<< endl << " "  << fundamental_mat << endl << endl;

    Mat K(3, 3, CV_64F);
    K.at<double>(0, 0) = 3310.400000;
    K.at<double>(0, 1) = 0.000000;
    K.at<double>(0, 2) = 316.730000;

    K.at<double>(1, 0) = 0.000000;
    K.at<double>(1, 1) = 3325.500000;
    K.at<double>(1, 2) = 200.550000;

    K.at<double>(2, 0) = 0.000000;
    K.at<double>(2, 1) = 0.000000;
    K.at<double>(2, 2) = 1.000000;

    Mat essential_mat = K.t()*fundamental_mat*K;

//    cout << "Essential Matrix = "<< endl << " "  << essential_mat << endl << endl;

    Mat rotation1, rotation2, translation;
    decomposeEssentialMat(essential_mat, rotation1, rotation2, translation);
    cout << "R1 Matrix = "<< endl << " "  << rotation1 << endl << endl;
    cout << "R2 Matrix = "<< endl << " "  << rotation2 << endl << endl;
    cout << "T Matrix = "<< endl << " "  << translation << endl << endl;

    float total_err = 0;
    for(int i=0; i<point_count; i++){
        Mat p1(3, 1, CV_64F);
        p1.at<double>(0, 0) = points1[i].x;
        p1.at<double>(1, 0) = points1[i].y;
        p1.at<double>(2, 0) = 1;
        Mat p2(3, 1, CV_64F);
        p2.at<double>(0, 0) = points2[i].x;
        p2.at<double>(1, 0) = points2[i].y;
        p2.at<double>(2, 0) = 1;

        Mat res = (T1.inv()*p1).t() * fundamental_mat * (T2.inv()*p2);

        total_err += abs(res.at<double>(0,0));
    }

//    cout << total_err/view_range << endl;

    //-- Draw only "good" matches
    Mat img_matches;
    drawMatches( img_1, keypoints_1,
                 img_2, keypoints_2,
                 good_matches,
                 img_matches, Scalar::all(-1), Scalar::all(-1),
                 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //-- Show detected matches
//    imshow( "Good Matches", img_matches );
//    imwrite("res.png", img_matches);
//    for( int i = 0; i < (int)good_matches.size(); i++ ) {
//        printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  -- Distant: %f \n",
//              i, good_matches[i].queryIdx, good_matches[i].trainIdx, good_matches[i].distance);
//    }

    bundleAdjustmentForTwoViews(points1, points2, rotation1, rotation2, translation, K);
    display(argc,argv); //for test
    waitKey(0);
    return 0;
}
