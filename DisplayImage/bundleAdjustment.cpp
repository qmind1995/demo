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
    points3D.resize(67);
    for(int i =0 ; i< 67; i++) {

        //change point3d data structure from matrix -> vector
        points3D[i] = cv::Point3d(X.at<double>(0, i), X.at<double>(1, i), X.at<double>(2, i));
    }
    cout<<"3D Point matrix : \n"<<points3D<<endl;
    cout<<"debug";
    return points3D;
}

vector< cv::Point3d > bundleAdjustmentForTwoViews(vector<Point2d> points0,
                                 vector<Point2d> points1,
                                 Mat rotation1,
                                 Mat rotation2,
                                 Mat translation,
                                 Mat K
){
    //cal projection matrix
    Mat P1(3, 4, CV_64F);
    hconcat( K*rotation1, K*(cv::Mat::eye(3,3,CV_64FC1)), P1 );
    Mat P2(3, 4, CV_64F);
    hconcat( K*rotation2, K*translation, P1 );

    cout<< P1<<endl;
    cout<< K<<endl;

    //hot fix:
    int N= 67, NCAMS =2;


    std::vector< cv::Point3d > points3D;
    points3D = triAngulationForTwoViews(K,rotation2, translation,points0, points1, N);
    std::vector< std::vector< cv::Point2d > > pointsImg;
    pointsImg.resize(NCAMS);
    for(int i=0; i<NCAMS; i++){
        pointsImg[i].resize(N);
    }
    for(int i =0 ; i< N; i++){
        pointsImg[0][i] = points0[i];
        pointsImg[1][i] = points1[i];
    }
    std::vector< std::vector< int > > visibility;
    visibility.resize(NCAMS);
    for(int i=0; i<NCAMS; i++)  {
        visibility[i].resize(N);
        for(int j=0; j<N; j++){
            visibility[i][j]=1;
        }
    }
    // fill distortion (assume no distortion)
    std::vector< cv::Mat > distCoeffs;
    distCoeffs.resize(NCAMS);
    for(int i=0; i<NCAMS; i++) {
        distCoeffs[i] = cv::Mat(5,1,CV_64FC1, cv::Scalar::all(0));
    }

    // cameras intrinsic matrix
    std::vector< cv::Mat > cameraMatrix;
    cameraMatrix.resize(NCAMS);
    for(int i=0; i<NCAMS; i++) {
        cameraMatrix[i] =K;
    }

//    cvsba::Sba sba;
//    sba.run(points3D,  pointsImg,  visibility,  cameraMatrix,  R,  T, distCoeffs);

    std::vector<cv::Mat> Ks, dist_coeffs, Rs, ts;
    Ks.resize(NCAMS, K);
    dist_coeffs.resize(NCAMS, cv::Mat::zeros(5, 1, CV_64F));
    Rs.push_back(cv::Mat::eye(3, 3, CV_64F));// R for the first camera (index: 0)
    ts.push_back(cv::Mat::zeros(3, 1, CV_64F));// t for the first camera (index: 0)


    cv::Mat F = cv::findFundamentalMat(points0, points1, cv::FM_8POINT);
    cv::Mat E = K.t() * F * K;
    cv::Mat R, t;
    cv::recoverPose(E, points0, points1, K, R, t);
    Rs.push_back(R);// R for the second camera
    ts.push_back(t);// t for the second camera

    cout<<"R:  \n"<<R<<endl;
    cout<<"debug";
//    for (int c = 0; c < .cols; c++)
    cvsba::Sba sba;
    cvsba::Sba::Params param;
    param.type = cvsba::Sba::MOTIONSTRUCTURE;
    param.fixedIntrinsics = 5;
    param.fixedDistortion = 5;
    param.verbose = true;
    sba.setParams(param);
    sba.run(points3D, pointsImg, visibility, Ks, Rs, ts, dist_coeffs);
    cout<<"Initial error="<<sba.getInitialReprjError()<<". Final error="<<sba.getFinalReprjError()<<std::endl;
    cout<<"debug";
    return points3D;
}

/**
 *
 * @param nviews
 * @param tray_2DPts
 * @param visibility
 * @param Ks
 * @param Rs
 * @param Ts
 * @param dist_coeffs
 * @param E
 * @param F
 * @return
 */
vector<cv::Point3D> bundleAdjustmentForMultiViews(vector<std::vector<cv::Point2d> > tray_2DPts,
                                                  vector<cv::Point3d> point_3Ds,
                                                  vector<std::vector<int> >visibility,
                                                  vector<cv::Mat> Ks,
                                                  vector<cv::Mat> Rs,
                                                  vector<cv::Mat> Ts,
                                                  vector<vc::Mat> dist_coeffs,
                                                  Mat E,
                                                  Mat F){

    try{
        cvsba::Sba sba;
        cvsba::Sba::Params param;
        param.type = cvsba::Sba::MOTIONSTRUCTURE;
        param.fixedIntrinsics = 5;
        param.fixedDistortion = 5;
        param.verbose = true;
        sba.setParams(param);
        double error = sba.run(point_3Ds, tray_2DPts, visibility, Ks, Rs, Ts, dist_coeffs);
    }
    catch (cv::Exception) { }
}