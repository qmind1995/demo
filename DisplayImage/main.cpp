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
#include <cvsba/cvsba.h>

#include "flann/flann.h"
#include "flann/algorithms/lsh_index.h"
#include "bundleAdjustment.cpp"
//#include "get3Dpoints.cpp"
#include "nViewStructureFromMotion.cpp"
#include <fstream>

using namespace std;
using namespace cv;
void readme();
/*
 * @function main
 * @brief Main function
 */

//return T matrix
//Mat normalization_fundamental(vector<Point2d> points, vector<Point2d>& points_nor){
//    unsigned long len = points.size();
//    float x_centroid = 0;
//    float y_centroid = 0;
//    for(int i=0; i<len; i++){
//        x_centroid += points[i].x;
//        y_centroid += points[i].y;
//    }
//    x_centroid = x_centroid/len;
//    y_centroid = y_centroid/len;
//
//    for(int i=0; i<len; i++){
//        points_nor[i].x = points[i].x - x_centroid;
//        points_nor[i].y = points[i].y - y_centroid;
//    }
//
//    float average_distant_from_origin = 0;
//    for(int i=0; i<len; i++){
//        average_distant_from_origin += sqrt(points_nor[i].x*points_nor[i].x + points_nor[i].y*points_nor[i].y);
//    }
//    average_distant_from_origin = average_distant_from_origin/len;
//
//    for(int i=0; i<len; i++){
//        points_nor[i].x = points_nor[i].x / (average_distant_from_origin / sqrt(2.0));
//        points_nor[i].y = points_nor[i].y / (average_distant_from_origin / sqrt(2.0));
//    }
//
//    Mat res(3, 3, CV_64F);
//    res.at<double>(0, 0) = sqrt(2) / average_distant_from_origin;
//    res.at<double>(0, 1) = 0;
//    res.at<double>(0, 2) = - x_centroid * sqrt(2) / average_distant_from_origin;
//
//    res.at<double>(1, 0) = 0;
//    res.at<double>(1, 1) = sqrt(2) / average_distant_from_origin;
//    res.at<double>(1, 2) = - y_centroid * sqrt(2) / average_distant_from_origin;
//
//    res.at<double>(2, 0) = 0;
//    res.at<double>(2, 1) = 0;
//    res.at<double>(2, 2) = 1;
//
//    return  res;
//}

//cv::Mat rot2euler(const cv::Mat & rotationMatrix) {
//    cv::Mat euler(3,1,CV_64F);
//
//    double m00 = rotationMatrix.at<double>(0,0);
//    double m02 = rotationMatrix.at<double>(0,2);
//    double m10 = rotationMatrix.at<double>(1,0);
//    double m11 = rotationMatrix.at<double>(1,1);
//    double m12 = rotationMatrix.at<double>(1,2);
//    double m20 = rotationMatrix.at<double>(2,0);
//    double m22 = rotationMatrix.at<double>(2,2);
//
//    double x, y, z;
//
//    // Assuming the angles are in radians.
//    if (m10 > 0.998) { // singularity at north pole
//        x = 0;
//        y = CV_PI/2;
//        z = atan2(m02,m22);
//    }
//    else if (m10 < -0.998) { // singularity at south pole
//        x = 0;
//        y = -CV_PI/2;
//        z = atan2(m02,m22);
//    }
//    else
//    {
//        x = atan2(-m12,m11);
//        y = asin(m10);
//        z = atan2(-m20,m00);
//    }
//
//    euler.at<double>(0) = x;
//    euler.at<double>(1) = y;
//    euler.at<double>(2) = z;
//
//    return euler;
//}

//int main( int argc, char** argv )
//{
//    string image_1_address = "./pictures/dinoRing/dinoR0001.png";//argv[1];
//    string image_2_address = "./pictures/dinoRing/dinoR0002.png";//argv[2];
//    Mat img_1 = imread( image_1_address, IMREAD_GRAYSCALE );
//    Mat img_2 = imread( image_2_address, IMREAD_GRAYSCALE );
//
//    //Detect the keypoints using Detector, compute the descriptors
//    Ptr<BRISK> detector = BRISK::create();
//
//    std::vector<KeyPoint> keypoints_1, keypoints_2;
//    Mat descriptors_1, descriptors_2;
//    detector->detectAndCompute( img_1, Mat(), keypoints_1, descriptors_1 );
//    detector->detectAndCompute( img_2, Mat(), keypoints_2, descriptors_2 );
//
//    //Matching descriptor vectors using FLANN matcher and check symmetric
//    cv::Ptr<cv::DescriptorMatcher> matcher = cv::makePtr<cv::FlannBasedMatcher>(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
//    std::vector< DMatch > matchesOneToTwo;
//    std::vector< DMatch > matchesTwoToOne;
//    std::vector< DMatch > matches;
//    matcher->match( descriptors_1, descriptors_2, matchesOneToTwo, Mat() );
//    matcher->match( descriptors_2, descriptors_1, matchesTwoToOne, Mat() );
//
//    for(int i = 0; i < descriptors_1.rows; i++){
//        if(matchesTwoToOne[matchesOneToTwo[i].trainIdx].trainIdx == i){
//            matches.push_back(DMatch(i, matchesOneToTwo[i].trainIdx, matchesOneToTwo[i].distance));
//        }
//    }
//
//
//    //sort match
//    sort(matches.begin(), matches.end());
//
//    int view_range = min((int)matches.size(), 100);
//    std::vector< DMatch > good_matches;
//    for(int i=0; i<view_range; i++)
//        good_matches.push_back(matches[i]);
//
//    unsigned long point_count = (unsigned long)view_range;
//    vector<Point2d> points1(point_count);
//    vector<Point2d> points2(point_count);
//
//    for(int i=0; i<point_count; i++){
//        float x1 = keypoints_1[good_matches[i].queryIdx].pt.x;
//        float y1 = keypoints_1[good_matches[i].queryIdx].pt.y;
//
//        float x2 = keypoints_2[good_matches[i].trainIdx].pt.x;
//        float y2 = keypoints_2[good_matches[i].trainIdx].pt.y;
//
//        points1[i] = Point2d(x1, y1);
//        points2[i] = Point2d(x2, y2);
//    }
//
//    vector<Point2d> points1_nor(point_count);
//    vector<Point2d> points2_nor(point_count);
//
//    Mat T1 = normalization_fundamental(points1, points1_nor);
//    Mat T2 = normalization_fundamental(points2, points2_nor);
//
//    Mat fundamental_mat = findFundamentalMat(points1_nor, points2_nor, FM_RANSAC, 3, 0.99);
//
//    fundamental_mat = T1.t()*fundamental_mat*T2;
//
////    cout << "T1 Matrix = "<< endl << " "  << T1 << endl << endl;
////    cout << "T2 Matrix = "<< endl << " "  << T2 << endl << endl;
////    cout << "Fundamental Matrix = "<< endl << " "  << fundamental_mat << endl << endl;
//
//    Mat K(3, 3, CV_64F);
//    K.at<double>(0, 0) = 3310.400000;
//    K.at<double>(0, 1) = 0.000000;
//    K.at<double>(0, 2) = 316.730000;
//
//    K.at<double>(1, 0) = 0.000000;
//    K.at<double>(1, 1) = 3325.500000;
//    K.at<double>(1, 2) = 200.550000;
//
//    K.at<double>(2, 0) = 0.000000;
//    K.at<double>(2, 1) = 0.000000;
//    K.at<double>(2, 2) = 1.000000;
//
//    Mat essential_mat = K.t()*fundamental_mat*K;
//
////    cout << "Essential Matrix = "<< endl << " "  << essential_mat << endl << endl;
//
//    Mat rotation1, rotation2, translation;
//    decomposeEssentialMat(essential_mat, rotation1, rotation2, translation);
//    cout << "R1 Matrix = "<< endl << " "  << rotation1 << endl << endl;
//    cout << "R2 Matrix = "<< endl << " "  << rotation2 << endl << endl;
//    cout << "T Matrix = "<< endl << " "  << translation << endl << endl;
//
//    float total_err = 0;
//    for(int i=0; i<point_count; i++){
//        Mat p1(3, 1, CV_64F);
//        p1.at<double>(0, 0) = points1[i].x;
//        p1.at<double>(1, 0) = points1[i].y;
//        p1.at<double>(2, 0) = 1;
//        Mat p2(3, 1, CV_64F);
//        p2.at<double>(0, 0) = points2[i].x;
//        p2.at<double>(1, 0) = points2[i].y;
//        p2.at<double>(2, 0) = 1;
//
//        Mat res = p1.t() * fundamental_mat * p2;
//
//        total_err += abs(res.at<double>(0,0));
//    }
//
////    cout << total_err/view_range << endl;
//
//    //-- Draw only "good" matches
//    Mat img_matches;
//    drawMatches( img_1, keypoints_1,
//                 img_2, keypoints_2,
//                 good_matches,
//                 img_matches, Scalar::all(-1), Scalar::all(-1),
//                 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
//    //-- Show detected matches
////    imshow( "Good Matches", img_matches );
////    imwrite("res.png", img_matches);
////    for( int i = 0; i < (int)good_matches.size(); i++ ) {
////        printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  -- Distant: %f \n",
////              i, good_matches[i].queryIdx, good_matches[i].trainIdx, good_matches[i].distance);
////    }
//
////    std::vector< cv::Point3d > point3ds=  bundleAdjustmentForTwoViews(points1, points2, rotation1, rotation2, translation, K);
////    display(argc,argv,point3ds); //for test
//    waitKey(0);
//    return 0;
//}

void writeMeshLabFile(string fileName,vector< cv::Point3d > points3D){

    ofstream plyFile (fileName);

    //write header:
    plyFile <<"ply"<<endl
            <<"format ascii 1.0"<<endl;
    plyFile <<"element vertex "<<points3D.size()<<endl;
    plyFile <<"property float32 x\n"
            <<"property float32 y\n"
            <<"property float32 z\n"
            <<"element face 0\n"
            <<"property list uint8 int32 vertex_indices\n"
            <<"end_header\n";
    for(int i=0; i<points3D.size();i++){
        plyFile<< points3D[i].x <<" "<< points3D[i].y <<" "<<points3D[i].z<<endl;
    }
    plyFile.close();
}

//void extractPairImageInfo(
//        const vector<string> image_address,
//        const Mat K,
//        const int ceilViewRange = 100,
//        vector<Point3d>& points,
//        vector<vector<Point2d>>& imagePoints,
//        vector<vector<int>>& visibility,
//        vector<Mat>& R_global,
//        vector<Mat>& T_global);

int main( int argc, char** argv ){

    Mat K(3, 3, CV_64F);
    K.at<double>(0, 0) = 2905.88;
    K.at<double>(0, 1) = 0.000000;
    K.at<double>(0, 2) = 1416;

    K.at<double>(1, 0) = 0.000000;
    K.at<double>(1, 1) = 2905.88;
    K.at<double>(1, 2) = 1064;

    K.at<double>(2, 0) = 0.000000;
    K.at<double>(2, 1) = 0.000000;
    K.at<double>(2, 2) = 1.000000;
    bool showMatching = true;

    int numImageTest = 4;

    string imageFolder = "./pictures/ImageDataset_SceauxCastle-master/images/";
    string imageListFile = "./pictures/ImageDataset_SceauxCastle-master/images/list_name.txt";
    vector<string> imageList;
    imageList.resize(numImageTest);
    ifstream imgListFileStream (imageListFile);

    if(! imgListFileStream.is_open()){
        return -1;
    }

    for(int i=0; i< numImageTest; i++){
        imgListFileStream >> imageList[i];
        imageList[i] = imageFolder + imageList[i];
    }

    vector< cv::Point3d > points3D;
    vector<vector<Point2d>> imagePoints;
    vector<vector<int>> visibility;
    vector<Mat> R_global;
    vector<Mat> T_global;
    extractPairImageInfo(imageList, K, 100, points3D, imagePoints, visibility, R_global, T_global);

    vector<Mat> dist_coeffs, Ks;
    Ks.resize((unsigned long) numImageTest, K);
    dist_coeffs.resize((unsigned long) numImageTest, Mat::zeros(5, 1, CV_64F));
    cvsba::Sba sba;
    cvsba::Sba::Params param;
    param.type = cvsba::Sba::MOTIONSTRUCTURE;
    param.fixedIntrinsics = 5;
    param.fixedDistortion = 5;
    param.verbose = false;
    sba.setParams(param);
    sba.run(points3D, imagePoints, visibility, Ks, R_global, T_global, dist_coeffs);


//    vector< cv::Point3d > points3D;
//    for(int i = 0 ; i< numImageTest-1;i++){
//        int image0_index = i;
//        int image1_index = i + 1;
//        cout << image0_index << endl;
//        string image0 = imageFolder + imageList[image0_index];
//        string image1 = imageFolder + imageList[image1_index];
//
//        vector<cv::Point3d> solvedPoints = get3DPointsFromTwoImg(image0, image1, K, showMatching, 200);
//        for (int j = 0; j < solvedPoints.size(); j++) {
//            points3D.push_back(solvedPoints[j]);
//        }
//    }



    writeMeshLabFile("test.ply",points3D);

    waitKey(0);
    return 0;
}
