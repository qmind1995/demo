//
// Created by tri on 22/04/2017.
//
#include <opencv/cv.hpp>
#include <cvsba/cvsba.h>
using namespace std;
using namespace cv;

//vector< cv::Point3d > triAngulationForTwoViews(Mat K, Mat R, Mat T,
//                                               vector<Point2d> points0,
//                                               vector<Point2d> points1, int N){
//    Mat P0 = K * cv::Mat::eye(3, 4, CV_64F);
//    Mat Rt, X;
//    hconcat(R, T, Rt);
//    Mat P1 = K * Rt;
//    cv::triangulatePoints(P0, P1, points0, points1, X);
//
//    X.row(0) = X.row(0) / X.row(3);
//    X.row(1) = X.row(1) / X.row(3);
//    X.row(2) = X.row(2) / X.row(3);
//    X.row(3) = 1;
//
//    std::vector< cv::Point3d > points3D;
//    points3D.resize(N);
//    for(int i =0 ; i< N; i++) {
//        //change point3d data structure from matrix -> vector
//        points3D[i] = cv::Point3d(X.at<double>(0, i), X.at<double>(1, i), X.at<double>(2, i));
//    }
//    return points3D;
//}

vector< cv::Point3d >triAngulationForTwoViews(Mat K, Mat R0, Mat T0,
                                              Mat R1, Mat T1,
                                              vector<Point2d> points0,
                                              vector<Point2d> points1, int N) {
    Mat RT0;
    hconcat(R0, T0, RT0);
    Mat P0 = K * RT0;

    Mat RT1;
    hconcat(R1, T1, RT1);
    Mat P1 = K * RT1;

    Mat X;
    triangulatePoints(P0, P1, points0, points1, X);

    X.row(0) = X.row(0) / X.row(3);
    X.row(1) = X.row(1) / X.row(3);
    X.row(2) = X.row(2) / X.row(3);
    X.row(3) = 1;
    std::vector< cv::Point3d > points3D;
    points3D.resize(N);
    for(int i =0 ; i< N; i++) {
        //change point3d data structure from matrix -> vector
        points3D[i] = cv::Point3d(X.at<double>(0, i), X.at<double>(1, i), X.at<double>(2, i));
    }
    return points3D;
}

/**
 *
 * @param imgPoints 2d point track
 * @param point3Ds
 * @param Rs
 * @param Ts
 * @param K
 * @param visibility
 * @param nview number of view
 * @return
 */
bool BAForMultiViews(vector< vector<Point2d> > imgPoints, vector<Point3d> point3Ds,
                                vector<Mat> &Rs, vector<Mat> &Ts, Mat K, vector< std::vector<int> > visibility, int nview){


    vector<Mat> Ks, dist_coeffs;
    Ks.resize(nview, K);
    dist_coeffs.resize(nview, Mat::zeros(5, 1, CV_64F));
    cvsba::Sba sba;
    cvsba::Sba::Params param;
    param.fixedIntrinsics = 5;
    param.fixedDistortion = 5;
    param.verbose = false;
    sba.setParams(param);
    try {
        sba.run(point3Ds, imgPoints, visibility, Ks, Rs, Ts, dist_coeffs);
        // add RS, TS to file
        return true;//success get 3d points and camera pose
    }
    catch (cv::Exception) {
        // need more matching
        return false;
    }

}