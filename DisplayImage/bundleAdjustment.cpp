//
// Created by tri on 22/04/2017.
//
#include <opencv/cv.hpp>
#include <cvsba/cvsba.h>
using namespace std;
using namespace cv;

vector< cv::Point3d > triAngulationForTwoViews(Mat K, Mat R, Mat T,
                                               vector<Point2d> points0,
                                               vector<Point2d> points1, int N){
    Mat P0 = K * cv::Mat::eye(3, 4, CV_64F);
    Mat Rt, X;
    hconcat(R, T, Rt);
    Mat P1 = K * Rt;
    cv::triangulatePoints(P0, P1, points0, points1, X);
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

vector< cv::Point3d > bundleAdjustmentForTwoViews(vector<Point2d> points0, vector<Point2d> points1,
                                 Mat rotation, Mat translation, Mat K, int N, int N_VIEWS =2){
    //cal projection matrix
    Mat P0 = K * Mat::eye(3, 4, CV_64F);
    Mat P1(3, 4, CV_64F);
    hconcat( K*rotation, K*translation, P0 );

    //hot fix:
    vector< Point3d > points3D;
    points3D = triAngulationForTwoViews(K,rotation, translation,points0, points1, N);
    vector< vector< Point2d > > pointsImg;
    pointsImg.resize(N_VIEWS);
    for(int i = 0; i < N_VIEWS; i++){
        pointsImg[i].resize(N);
    }
    for(int i = 0 ; i < N; i++){
        pointsImg[0][i] = points0[i];
        pointsImg[1][i] = points1[i];
    }
    vector< std::vector<int> > visibility;
    visibility.resize(N_VIEWS);
    for(int i=0; i<N_VIEWS; i++){
        visibility[i].resize(N);
        for(int j=0; j<N; j++){
            visibility[i][j]=1;
        }
    }
    // fill distortion (assume no distortion)
    vector< Mat > distCoeffs;
    distCoeffs.resize(N_VIEWS);
    for(int i=0; i<N_VIEWS; i++) {
        distCoeffs[i] = Mat(5,1,CV_64FC1, Scalar::all(0));
    }

    // cameras intrinsic matrix
    vector< Mat > cameraMatrix;
    cameraMatrix.resize(N_VIEWS);
    for(int i=0; i<N_VIEWS; i++) {
        cameraMatrix[i] =K;
    }

    vector<Mat> Ks, dist_coeffs, Rs, ts;
    Ks.resize(N_VIEWS, K);
    dist_coeffs.resize(N_VIEWS, Mat::zeros(5, 1, CV_64F));
    Rs.push_back(Mat::eye(3, 3, CV_64F));// R for the first camera (index: 0)
    ts.push_back(Mat::zeros(3, 1, CV_64F));// t for the first camera (index: 0)

    Rs.push_back(rotation);// R for the second camera
    ts.push_back(translation);// t for the second camera

    cvsba::Sba sba;
    cvsba::Sba::Params param;
    param.type = cvsba::Sba::MOTIONSTRUCTURE;
    param.fixedIntrinsics = 5;
    param.fixedDistortion = 5;
    param.verbose = false;
    sba.setParams(param);
    sba.run(points3D, pointsImg, visibility, Ks, Rs, ts, dist_coeffs);
    cout<<"Initial error="<<sba.getInitialReprjError()<<". Final error="<<sba.getFinalReprjError()<<std::endl;
    return points3D;
}