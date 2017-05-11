//
// Created by quang on 10/05/2017.
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
#include <opencv/cv.hpp>
#include <cvsba/cvsba.h>

using namespace std;
using namespace cv;

//get correspond point of 2 image
vector<vector<Point2d>> getPairsPoint(string image_1_address, string image_2_address, int ceilViewRange);
//calculate Rotation and Translation matrix, and refine correspond point of 2 image by remove outliner
//R, T is local
void calculateRTandReinfePairsPoint(vector<vector<Point2d>>& pairsPoint,
                                    Mat K, Mat& R, Mat& T);
//get 3D point generate by 2 image
//R, T use in this is global
vector<Point3d> get3Dpoints(vector<vector<Point2d>> pairsPoint,
                            Mat K1, Mat R1, Mat T1,
                            Mat K2, Mat R2, Mat T2);
//build track 3D point <=> vector of Point2d
vector<vector<Point3d>> buildTrack(vector<vector<vector<Point2d>>> pairsPointOfMultiImage);

//build visibility matrix and 3Dpoint location on each image matrix
void build3DpointOnImage(vector<vector<Point3d>> track,
                         int number_image,
                         vector<vector<Point2d>>& imagePoints,
                         vector<vector<int>>& visibility);


void get_R_T_from_RT(Mat& R, Mat& T, const Mat RT){
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            R.at<double>(i,j) = RT.at<double>(i,j);
        }
    }

    for(int i=0; i<3; i++){
        T.at<double>(i,0) = RT.at<double>(i,3);
    }
}

void get_RT_from_R_T(const Mat R, const Mat T, Mat& RT){
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            RT.at<double>(i,j) = R.at<double>(i,j);
        }
    }
    for(int i=0; i<3; i++){
        RT.at<double>(i,3) = T.at<double>(i,0);
    }
}

void extractPairImageInfo(
        vector<string> image_address,
        Mat K,
        int ceilViewRange,
        vector<Point3d>& points3D,
        vector<vector<Point2d>>& imagePoints,
        vector<vector<int>>& visibility,
        vector<Mat>& R_global,
        vector<Mat>& T_global)
{
    int number_image = image_address.size();
    int number_pair = number_image - 1;
    Mat temp_R(3, 3, CV_64F);
    Mat temp_T(3, 1, CV_64F);
    vector<Mat> RT_global(number_image);
    RT_global[0] = Mat::eye(4, 4, CV_64F);
    get_R_T_from_RT(temp_R, temp_T, RT_global[0]);
    R_global.push_back(temp_R);
    T_global.push_back(temp_T);

    vector<vector<vector<Point2d>>> pairsPointOfMultiImage;
    for(int pairs_id = 0; pairs_id < number_pair; pairs_id++){
        vector<vector<Point2d>> pairs_point_pair_image =
                getPairsPoint(image_address[pairs_id],image_address[pairs_id+1], 100);

        pairsPointOfMultiImage.push_back(pairs_point_pair_image);

        Mat current_R, current_T;
        calculateRTandReinfePairsPoint(
                pairs_point_pair_image,
                K, current_R, current_T);

        Mat current_RT = Mat::eye(4, 4, CV_64F);
        get_RT_from_R_T(current_R, current_T, current_RT);
        RT_global[pairs_id+1] = current_RT*RT_global[pairs_id];
        get_R_T_from_RT(temp_R, temp_T, RT_global[pairs_id+1]);
        R_global.push_back(temp_R);
        T_global.push_back(temp_T);

        Mat R1(3, 3, CV_64F);
        Mat R2(3, 3, CV_64F);
        Mat T1(3, 1, CV_64F);
        Mat T2(3, 1, CV_64F);
        get_R_T_from_RT(R1, T1, RT_global[pairs_id]);
        get_R_T_from_RT(R2, T2, RT_global[pairs_id+1]);
        vector<Point3d> current_point3D = get3Dpoints(
                pairs_point_pair_image,
                K, R1, T1,
                K, R2, T2);
        for(int point3D_id = 0; point3D_id < current_point3D.size(); point3D_id++){
            points3D.push_back(current_point3D[point3D_id]);
        }
    }

    vector<vector<Point3d>> track = buildTrack(pairsPointOfMultiImage);
    build3DpointOnImage(track, number_image, imagePoints, visibility);
}

vector<vector<Point2d>> getPairsPoint(
        string image_1_address,
        string image_2_address,
        int ceilViewRange){
    Mat img_1 = imread( image_1_address, IMREAD_GRAYSCALE );
    Mat img_2 = imread( image_2_address, IMREAD_GRAYSCALE );

    Ptr<BRISK> detector = BRISK::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::makePtr<cv::FlannBasedMatcher>(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));

    vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    detector->detectAndCompute( img_1, Mat(), keypoints_1, descriptors_1 );
    detector->detectAndCompute( img_2, Mat(), keypoints_2, descriptors_2 );

    vector< DMatch > matchesOneToTwo;
    vector< DMatch > matchesTwoToOne;
    vector< DMatch > matches;
    matcher->match( descriptors_1, descriptors_2, matchesOneToTwo, Mat() );
    matcher->match( descriptors_2, descriptors_1, matchesTwoToOne, Mat() );

    for(int i = 0; i < descriptors_1.rows; i++){
        if(matchesTwoToOne[matchesOneToTwo[i].trainIdx].trainIdx == i){
            matches.push_back(DMatch(i, matchesOneToTwo[i].trainIdx, matchesOneToTwo[i].distance));
        }
    }
//    matches = matchesOneToTwo;
    sort(matches.begin(), matches.end());

    int view_range = min((int)matches.size(), ceilViewRange);
    std::vector< DMatch > good_matches;
    for(int i=0; i<view_range; i++)
        good_matches.push_back(matches[i]);

    unsigned long point_count = (unsigned long)view_range;
    vector<Point2d> points0(point_count);
    vector<Point2d> points1(point_count);

    for(int i=0; i<point_count; i++){
        float x1 = keypoints_1[good_matches[i].queryIdx].pt.x;
        float y1 = keypoints_1[good_matches[i].queryIdx].pt.y;

        float x2 = keypoints_2[good_matches[i].trainIdx].pt.x;
        float y2 = keypoints_2[good_matches[i].trainIdx].pt.y;

        points0[i] = Point2d(x1, y1);
        points1[i] = Point2d(x2, y2);
    }

    vector<vector<Point2d>> res;
    res.push_back(points0);
    res.push_back(points1);
    return res;
}

void calculateRTandReinfePairsPoint(vector<vector<Point2d>>& pairsPoint,
                                    Mat K, Mat& R, Mat& T){
    vector<Point2d> points0 = pairsPoint[0];
    vector<Point2d> points1 = pairsPoint[1];
    Mat mask;
    Mat fundamental_mat = findFundamentalMat(points0, points1, FM_RANSAC, 3, 0.99, mask);

    vector<Point2d> point0_filtered;
    vector<Point2d> point1_filtered;
    for(int j=0; j < points0.size(); j++){
        if((unsigned int)mask.at<char>(j) == 1){
            point0_filtered.push_back(points0[j]);
            point1_filtered.push_back(points1[j]);
            //match_filtered[pairs_id].push_back(good_matches[j]);
        }
    }

    Mat essential_mat = K.t()*fundamental_mat*K;
    //check again from here!!!!!
    recoverPose(essential_mat,
                point0_filtered, point1_filtered, K,
                R, T);

    pairsPoint.clear();
    pairsPoint.push_back(point0_filtered);
    pairsPoint.push_back(point1_filtered);
}

vector<Point3d> get3Dpoints(vector<vector<Point2d>> pairsPoint,
                            Mat K1, Mat R1, Mat T1,
                            Mat K2, Mat R2, Mat T2){
    int num_point = pairsPoint[0].size();
    Mat RT1;
    Mat RT2;
    hconcat(R1, T1, RT1);
    hconcat(R2, T2, RT2);
    Mat P1 = K1 *  RT1;
    Mat P2 = K2 * RT2;

    Mat homo_3d_point;
    triangulatePoints(P1, P2,
                      pairsPoint[0], pairsPoint[1],
                      homo_3d_point);
    homo_3d_point.row(0) = homo_3d_point.row(0) / homo_3d_point.row(3);
    homo_3d_point.row(1) = homo_3d_point.row(1) / homo_3d_point.row(3);
    homo_3d_point.row(2) = homo_3d_point.row(2) / homo_3d_point.row(3);
    homo_3d_point.row(3) = 1;

    vector<Point3d> res(pairsPoint[0].size());

    for(int i = 0 ; i< num_point; i++) {
        //change point3d data structure from matrix -> vector
        res[i] = Point3d(
                homo_3d_point.at<double>(0, i),
                homo_3d_point.at<double>(1, i),
                homo_3d_point.at<double>(2, i));
    }
    return res;
}

vector<vector<Point3d>> buildTrack(vector<vector<vector<Point2d>>> pairsPointOfMultiImage){
    int number_pair = pairsPointOfMultiImage.size();
    vector<vector<Point3d>> track;
    for(int pairs_id = 0; pairs_id < number_pair; pairs_id++){
        //for each pair of point
        int number_point_current_pair_image = pairsPointOfMultiImage[pairs_id][0].size();
        for(int pairs_point_id = 0; pairs_point_id < number_point_current_pair_image; pairs_point_id++){
            //for each exited 3d-point-chain track
            bool pair_exited = false;

            for(int item_id = 0; item_id < track.size(); item_id++){
                //int current_x = (int)key_point_vector[pairs_id][match_filtered[pairs_id][pairs_point_id].queryIdx].pt.x;
                //int current_y = (int)key_point_vector[pairs_id][match_filtered[pairs_id][pairs_point_id].queryIdx].pt.y;
                int current_x = pairs_id;
                int current_y = pairsPointOfMultiImage[pairs_id][0][pairs_point_id].x;
                int current_z = pairsPointOfMultiImage[pairs_id][0][pairs_point_id].y;

                bool x_equal = track[item_id].back().x == current_x;
                bool y_equal = track[item_id].back().y == current_y;
                bool z_equal = track[item_id].back().z == current_z;
                if(x_equal && y_equal && z_equal){
                    pair_exited = true;
                    track[item_id].push_back(
                            Point3d(pairs_id+1,
                                    pairsPointOfMultiImage[pairs_id][1][pairs_point_id].x,
                                    pairsPointOfMultiImage[pairs_id][1][pairs_point_id].y
                            ));
                    break;
                }
            }
            if(!pair_exited){
                //points3D.push_back(point3D_pair[pairs_id][pairs_point_id]);
                vector<Point3d> new_node;
                new_node.push_back(
                        Point3d(pairs_id,
                                pairsPointOfMultiImage[pairs_id][0][pairs_point_id].x,
                                pairsPointOfMultiImage[pairs_id][0][pairs_point_id].y
                        ));

                new_node.push_back(
                        Point3d(pairs_id+1,
                                pairsPointOfMultiImage[pairs_id][1][pairs_point_id].x,
                                pairsPointOfMultiImage[pairs_id][1][pairs_point_id].y));
                track.push_back(new_node);
            }
        }
    }
    return track;
}

void build3DpointOnImage(vector<vector<Point3d>> track,
                         int number_image,
                         vector<vector<Point2d>>& imagePoints,
                         vector<vector<int>>& visibility){
    int num_3D_points = track.size();

    for(int image_id = 0; image_id < number_image; image_id++){
        vector<Point2d> temp_vector_point2d;
        vector<int> temp_visibility;
        for(int point3d_id = 0; point3d_id < num_3D_points; point3d_id++){
            temp_vector_point2d.push_back(Point2d());
            temp_visibility.push_back(0);
        }
        imagePoints.push_back(temp_vector_point2d);
        visibility.push_back(temp_visibility);
    }

    //points3D.resize(num_3D_points);
    for(int point3d_id = 0; point3d_id < num_3D_points; point3d_id++){
        for(int item_id = 0; item_id < track[point3d_id].size(); item_id++){
            int image_id = (int)track[point3d_id][item_id].x;
            int key_point_x = (int)track[point3d_id][item_id].y;
            int key_point_y = (int)track[point3d_id][item_id].z;
            imagePoints[image_id][point3d_id].x = key_point_x;
            imagePoints[image_id][point3d_id].y = key_point_y;
            visibility[image_id][point3d_id] = 1;
        }
    }
}