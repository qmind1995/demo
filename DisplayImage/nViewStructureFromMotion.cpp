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
//#include <cublas_v2.h>

using namespace std;
using namespace cv;


/**
 * get correspond point of 2 image
 * @param image_1_address
 * @param image_2_address
 * @param ceilViewRange
 * @return
 */
vector<vector<Point2d>> getPairsPoint(string image_1_address, string image_2_address, int ceilViewRange);

/**
 * calculate Rotation and Translation matrix, and refine correspond point of 2 image by remove outliner.
 * R, T is local
 * @param pairsPoint
 * @param K
 * @param R
 * @param T
 */
void calculateRTandReinfePairsPoint(vector<vector<Point2d>>& pairsPoint,
                                    Mat K, Mat& R, Mat& T);

/**
 * get 3D point generate by 2 image
 * R, T use in this is global
 * @param pairsPoint
 * @param K
 * @param R
 * @param T
 * @return
 */
vector<Point3d> get3Dpoints(vector<vector<Point2d>> pairsPoint,
                            Mat K, Mat R, Mat T);

/**
 * build track 3D point <=> vector of Point2d
 * @param pairsPointOfMultiImage
 * @return
 */
vector<vector<Point2d>> buildTrack(vector<vector<vector<Point2d>>> pairsPointOfMultiImage);

/**
 * build visibility matrix and 3Dpoint location on each image matrix
 * @param track
 * @param imagePoints
 * @param visibility
 */
void build3DpointOnImage(vector<vector<Point2d>> track,
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
        const vector<string> image_address,
        const Mat K,
        const int ceilViewRange,
        vector<Point3d>& points3D,
        vector<vector<Point2d>>& imagePoints,
        vector<vector<int>>& visibility,
        vector<Mat>& R_global,
        vector<Mat>& T_global)
{
    Ptr<BRISK> detector = BRISK::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::makePtr<cv::FlannBasedMatcher>(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));

    unsigned long number_image = image_address.size();
    unsigned long number_pair = number_image - 1;

    R_global.resize(number_image);
    T_global.resize(number_image);
    for(unsigned long image_id = 0; image_id < number_image; image_id++){
        R_global[image_id] = Mat::zeros(3, 3, CV_64F);
        T_global[image_id] = Mat::zeros(3, 1, CV_64F);
    }

    //calculate key point and descriptor for each image
    vector<vector<KeyPoint>> key_point_vector(number_image);
    vector<Mat> descriptor_vector(number_image);

    for(unsigned long image_id = 0; image_id < number_image; image_id++){
        Mat current_image = imread( image_address[image_id], IMREAD_GRAYSCALE );
        detector->detectAndCompute(
                current_image, Mat(),
                key_point_vector[image_id],
                descriptor_vector[image_id] );
    }

    vector<vector<DMatch>> match_filtered(number_pair); //filter by RANSAC
    vector<vector<Point3d>> point3D_pair(number_pair);

    vector<Mat> RT_global(number_image);
    RT_global[0] = Mat::eye(4, 4, CV_64F);
    //Mat current_global_RT = Mat::eye(4, 4, CV_64F);
    get_R_T_from_RT(R_global[0], T_global[0], RT_global[0]);
    for(unsigned long pairs_id = 0; pairs_id < number_pair; pairs_id++){
        vector<DMatch> matchesOneToTwo;
        vector<DMatch> matchesTwoToOne;
        vector<DMatch> matches;
        matcher->match(
                descriptor_vector[pairs_id],
                descriptor_vector[pairs_id+1],
                matchesOneToTwo, Mat() );
        matcher->match(
                descriptor_vector[pairs_id+1],
                descriptor_vector[pairs_id],
                matchesTwoToOne, Mat() );

        for(int point_id = 0; point_id < descriptor_vector[pairs_id].rows; point_id++){
            if(matchesTwoToOne[matchesOneToTwo[point_id].trainIdx].trainIdx == point_id){
                matches.push_back(DMatch(point_id,
                               matchesOneToTwo[point_id].trainIdx,
                               matchesOneToTwo[point_id].distance));
            }
        }

        sort(matches.begin(), matches.end());
        vector<DMatch> good_matches;
        int watch_range = min((int)matches.size(), ceilViewRange);
        for(int j=0; j<watch_range; j++)
            good_matches.push_back(matches[j]);

        vector<Point2d> points0((unsigned long) watch_range);
        vector<Point2d> points1((unsigned long) watch_range);

        for(int point_id=0; point_id < watch_range; point_id++){
            float x1 = key_point_vector[pairs_id][good_matches[point_id].queryIdx].pt.x;
            float y1 = key_point_vector[pairs_id][good_matches[point_id].queryIdx].pt.y;

            float x2 = key_point_vector[pairs_id+1][good_matches[point_id].trainIdx].pt.x;
            float y2 = key_point_vector[pairs_id+1][good_matches[point_id].trainIdx].pt.y;

            points0[point_id] = Point2d(x1, y1);
            points1[point_id] = Point2d(x2, y2);
        }

        Mat mask;
        Mat fundamental_mat = findFundamentalMat(points0, points1, FM_RANSAC, 3, 0.99, mask);

        vector<Point2d> point0_filtered;
        vector<Point2d> point1_filtered;
        for(int j=0; j < points0.size(); j++){
            if((unsigned int)mask.at<char>(j) == 1){
                point0_filtered.push_back(points0[j]);
                point1_filtered.push_back(points1[j]);
                match_filtered[pairs_id].push_back(good_matches[j]);
            }
        }

        Mat essential_mat = K.t()*fundamental_mat*K;
        Mat current_pair_R;
        Mat current_pair_T;
        Mat current_pair_RT(4, 4, CV_64F);
        recoverPose(essential_mat,
                    point0_filtered, point1_filtered, K,
                    current_pair_R,
                    current_pair_T);

        get_RT_from_R_T(current_pair_R, current_pair_T, current_pair_RT);
        //X1 = P0*X0
        //X2 = P1*X1 => X2 = (P1*P0)*X0
        RT_global[pairs_id+1] = current_pair_RT*RT_global[pairs_id];
        //get global R,T for each image except first image
        get_R_T_from_RT(R_global[pairs_id+1], T_global[pairs_id+1], RT_global[pairs_id+1]);
        //we have R_global and T_global of image before!
        Mat homo_3d_point;
        cout << RT_global[0] << endl << endl;
        cout << RT_global[1] << endl << endl;
        Mat eye34 = cv::Mat::eye(3, 4, CV_64F);
        Mat Projection_0 = K * eye34 * RT_global[pairs_id];
        Mat Projection_1 = K * eye34 * RT_global[pairs_id+1];
        triangulatePoints(Projection_0, Projection_1,
                          point0_filtered, point1_filtered,
                          homo_3d_point);
        homo_3d_point.row(0) = homo_3d_point.row(0) / homo_3d_point.row(3);
        homo_3d_point.row(1) = homo_3d_point.row(1) / homo_3d_point.row(3);
        homo_3d_point.row(2) = homo_3d_point.row(2) / homo_3d_point.row(3);
        homo_3d_point.row(3) = 1;
        point3D_pair[pairs_id].resize(match_filtered.size());
        for(int i =0 ; i< watch_range; i++) {
            //change point3d data structure from matrix -> vector
            point3D_pair[pairs_id][i] = Point3d(
                    homo_3d_point.at<double>(0, i),
                    homo_3d_point.at<double>(1, i),
                    homo_3d_point.at<double>(2, i));
        }
    }

    //2D track
    //x for image index
    //y for point index in keypoint vector
    vector<vector<Point2d>> track;
    //for each pair of image
    for(int pairs_id = 0; pairs_id < number_pair; pairs_id++){
        //for each pair of point
        for(int pairs_point_id = 0; pairs_point_id < match_filtered[pairs_id].size(); pairs_point_id++){
            //for each exited 3d-point-chain track
            bool pair_exited = false;

            for(int item_id = 0; item_id < track.size(); item_id++){
                //int current_x = (int)key_point_vector[pairs_id][match_filtered[pairs_id][pairs_point_id].queryIdx].pt.x;
                //int current_y = (int)key_point_vector[pairs_id][match_filtered[pairs_id][pairs_point_id].queryIdx].pt.y;
                int current_x = pairs_id;
                int current_y = match_filtered[pairs_id][pairs_point_id].queryIdx;

                bool x_equal = track[item_id].back().x == current_x;
                bool y_equal = track[item_id].back().y == current_y;
                if(x_equal && y_equal){
                    pair_exited = true;
                    track[item_id].push_back(
                            Point2d(pairs_id+1,
                                    match_filtered[pairs_id][pairs_point_id].trainIdx));
                    break;
                }
            }
            if(!pair_exited){
                points3D.push_back(point3D_pair[pairs_id][pairs_point_id]);
                vector<Point2d> new_node;
                new_node.push_back(Point2d(pairs_id, match_filtered[pairs_id][pairs_point_id].queryIdx));
                new_node.push_back(Point2d(pairs_id+1, match_filtered[pairs_id][pairs_point_id].trainIdx));
                track.push_back(new_node);
            }
        }
    }

//    vector<Point3d>& points3D,
//    vector<vector<Point2d>>& imagePoints, (num camera) x (num 3d point)
//    vector<vector<int>>& visibility,
    unsigned long num_3D_points = track.size();
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

    points3D.resize(num_3D_points);
    for(int point3d_id = 0; point3d_id < num_3D_points; point3d_id++){
        for(int item_id = 0; item_id < track[point3d_id].size(); item_id++){
            int image_id = (int)track[point3d_id][item_id].x;
            int key_point_id = (int)track[point3d_id][item_id].y; //in key point vector
            imagePoints[image_id][point3d_id].x = key_point_vector[image_id][key_point_id].pt.x;
            imagePoints[image_id][point3d_id].y = key_point_vector[image_id][key_point_id].pt.y;
            visibility[image_id][point3d_id] = 1;
        }
    }
}

vector<vector<Point2d>> getPairsPoint(
        string image_1_address,
        string image_2_address,
        int ceilViewRange){
    Mat img_1 = imread( image_1_address, IMREAD_GRAYSCALE );
    Mat img_2 = imread( image_2_address, IMREAD_GRAYSCALE );

    Ptr<BRISK> detector = BRISK::create();
//    cv::Ptr<cv::DescriptorMatcher> matcher = cv::makePtr<cv::FlannBasedMatcher>(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));

    vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    detector->detectAndCompute( img_1, Mat(), keypoints_1, descriptors_2 );
    detector->detectAndCompute( img_2, Mat(), keypoints_1, descriptors_2 );

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::makePtr<cv::FlannBasedMatcher>(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
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