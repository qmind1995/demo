//
// Created by tri on 13/05/2017.
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
#include <cvsba/cvsba.h>
using namespace std;
using namespace cv;

vector<Point3d> translateWorld(Mat Rt, vector<Point3d> points3D){
    vector< cv::Point3d > Ps;
    for(int i =0; i< points3D.size(); i++){
        Mat point(4, 1, CV_64F);

        point.at<double>(0,0) = points3D[i].x;
        point.at<double>(1,0) = points3D[i].y;
        point.at<double>(2,0) = points3D[i].z;
        point.at<double>(3,0) = 1;
        Mat p = Rt* point;

        Ps.push_back(Point3d(p.at<double>(0, 0), p.at<double>(1, 0), p.at<double>(2, 0)));
    }
    return Ps;
}

vector<Point3d> translateOnly(Mat T, vector<Point3d> points3D){
    vector<Point3d> test;
    for(int i= 0; i<points3D.size(); i++){
        test.push_back(Point3d(points3D[i].x + T.at<double>(0,0),
                               points3D[i].y + T.at<double>(0,1),
                               points3D[i].z + T.at<double>(0,2) +2));
    }
    return test;
}

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

vector< cv::Point3d > bundleAdjustmentForTwoViews(
        vector<Point2d> points0,
        vector<Point2d> points1,
        Mat rotation,
        Mat translation,
        Mat K,
        int N,
        int N_VIEWS =2){
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
    cout<<"R1:"<<endl;
    cout<<Rs[0]<<endl;
    cout<<"T1:"<<endl;
    cout<<ts[0]<<endl;

    cout<<"R2:"<<endl;
    cout<<Rs[1]<<endl;
    cout<<"T2:"<<endl;
    cout<<ts[1]<<endl;

    Mat Rt_1 ,Rt_2;
//
//    if(points3D.size() > 88){
//        hconcat(Rs[0], ts[0], Rt_1);
//        vector<Point3d> test  =  translateWorld(Rt_1, points3D);
//        hconcat(Rs[1], ts[1], Rt_2);
//        vector<Point3d> test_ = translateWorld(Rt_2, points3D);
//
//        return test;
//    }
//    else{
//        hconcat(Rs[0], ts[0], Rt_1);
//        vector<Point3d> test  =  translateWorld(Rt_1, points3D);
//        return test;
//    }
    return points3D;
}

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
        float x1 = keypoints_0[good_matches[i].queryIdx].pt.x;
        float y1 = keypoints_0[good_matches[i].queryIdx].pt.y;

        float x2 = keypoints_1[good_matches[i].trainIdx].pt.x;
        float y2 = keypoints_1[good_matches[i].trainIdx].pt.y;

        points0[i] = Point2d(x1, y1);
        points1[i] = Point2d(x2, y2);
    }

    vector<Point2d> points0_nor(point_count);
    vector<Point2d> points1_nor(point_count);

    //Mat T1 = normalization_fundamental(points0, points0_nor);
    //Mat T2 = normalization_fundamental(points1, points1_nor);

//    Mat fundamental_mat = findFundamentalMat(points0_nor, points1_nor, FM_RANSAC, 3, 0.99);
//    fundamental_mat = T1.t()*fundamental_mat*T2;

    Mat mask;

    Mat fundamental_mat = findFundamentalMat(points0, points1, FM_RANSAC, 3, 0.99, mask);

    std::vector< DMatch > good_matches_filted;
    vector<double> err;
    vector<Point2d> point0_fileted;
    vector<Point2d> point1_fileted;
    double total_err_not_de_norm = 0;
    double count_err = 0;
    for(int i=0; i < points0.size(); i++){
        Mat p1(3, 1, CV_64F);
        p1.at<double>(0, 0) = points0[i].x;
        p1.at<double>(1, 0) = points0[i].y;
        p1.at<double>(2, 0) = 1;
        Mat p2(3, 1, CV_64F);
        p2.at<double>(0, 0) = points1[i].x;
        p2.at<double>(1, 0) = points1[i].y;
        p2.at<double>(2, 0) = 1;

        Mat epipola_line_1 = fundamental_mat * p2;
        Mat res_up_mat_1 = p1.t() * epipola_line_1;
        double res_up_1 = abs(res_up_mat_1.at<double>(0,0));
        double res_bot_1 = sqrt(pow(epipola_line_1.at<double>(0,0),2) + pow(epipola_line_1.at<double>(1,0),2));
        double distant_1 = res_up_1/res_bot_1;

        Mat epipola_line_2 = p1.t() * fundamental_mat;
        Mat res_up_mat_2 = epipola_line_2 * p2;
        double res_up_2 = abs(res_up_mat_2.at<double>(0,0));
        double res_bot_2 = sqrt(pow(epipola_line_2.at<double>(0,0),2) + pow(epipola_line_2.at<double>(0,1),2));
        double distant_2 = res_up_2/res_bot_2;
        //Mat res = p1.t() * fundamental_mat * p2;
//        cout << (unsigned int)mask.at<char>(i) << "---" << max(distant_1, distant_2) << endl << endl;

        err.push_back(max(distant_1, distant_2));

        if((unsigned int)mask.at<char>(i) == 1){
            total_err_not_de_norm += max(distant_1, distant_2);
            count_err += 1;
            point0_fileted.push_back(points0[i]);
            point1_fileted.push_back(points1[i]);
            good_matches_filted.push_back(good_matches[i]);
        }
    }



//    cout << "Total fundamental err not de-norm yet: " << total_err_not_de_norm/count_err << endl << endl;
//    cout << fundamental_mat << endl;
    Mat essential_mat = K.t()*fundamental_mat*K;

    Mat rotation, translation;
    recoverPose(essential_mat, point0_fileted, point1_fileted, K, rotation, translation);

    if(showMatching){
        Mat img_matches;
        drawMatches( img_1, keypoints_0,
                     img_2, keypoints_1,
                //good_matches,
                     good_matches_filted,
                     img_matches, Scalar::all(-1), Scalar::all(-1),
                     vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        imshow( "Good Matches", img_matches );
        imwrite("res.png", img_matches);
    }

    vector< cv::Point3d > points3D = bundleAdjustmentForTwoViews(point0_fileted, point1_fileted, rotation, translation, K, point0_fileted.size());

    return points3D;
}

vector< DMatch > matchingTwoViews(vector<KeyPoint> keypoints_0,vector<KeyPoint> keypoints_1,
                                  Mat descriptors_1, Mat descriptors_2,
                                  Ptr<cv::DescriptorMatcher> matcher){
    vector< DMatch > matchesOneToTwo;
    vector< DMatch > matchesTwoToOne;
    vector<DMatch> matches;
    matcher->match( descriptors_1, descriptors_2, matchesOneToTwo, Mat() );
    matcher->match( descriptors_2, descriptors_1, matchesTwoToOne, Mat() );

    for(int i = 0; i < descriptors_1.rows; i++){
        if(matchesTwoToOne[matchesOneToTwo[i].trainIdx].trainIdx == i){
            matches.push_back(DMatch(i, matchesOneToTwo[i].trainIdx, matchesOneToTwo[i].distance));
        }
    }
    sort(matches.begin(), matches.end());
    int view_range = min((int)matches.size(), 100);
    vector< DMatch > goodMatch;
    for(int i=0; i<view_range; i++){
        goodMatch.push_back(matches[i]);
    }
    return goodMatch;
}

vector<Point2d> get2DPoints(vector< DMatch > matches, vector<KeyPoint> keypoints, bool isFirst){
    int view_range = matches.size();
    vector<Point2d> points;

    for(int i=0; i<view_range; i++){
        if(isFirst == true){
            float x1 = keypoints[matches[i].queryIdx].pt.x;
            float y1 = keypoints[matches[i].queryIdx].pt.y;
            points.push_back(Point2d(x1, y1)) ;
        }
        else{
            float x1 = keypoints[matches[i].trainIdx].pt.x;
            float y1 = keypoints[matches[i].trainIdx].pt.y;
            points.push_back(Point2d(x1, y1)) ;
        }
    }
    return points;
}

/**
 * refine matching and return fundamental matrix
 * @return
 */
Mat refineMatching(vector<Point2d> & points0, vector<Point2d> & points1, vector< DMatch > & good_matches){
    Mat mask;
    Mat fundamental_mat = findFundamentalMat(points0, points1, FM_RANSAC, 3, 0.99, mask);

    std::vector< DMatch > good_matches_filted;
    vector<double> err;
    vector<Point2d> point0_fileted;
    vector<Point2d> point1_fileted;
    double total_err_not_de_norm = 0;
    double count_err = 0;
    for(int i=0; i < points0.size(); i++){
        Mat p1(3, 1, CV_64F);
        p1.at<double>(0, 0) = points0[i].x;
        p1.at<double>(1, 0) = points0[i].y;
        p1.at<double>(2, 0) = 1;
        Mat p2(3, 1, CV_64F);
        p2.at<double>(0, 0) = points1[i].x;
        p2.at<double>(1, 0) = points1[i].y;
        p2.at<double>(2, 0) = 1;

        Mat epipola_line_1 = fundamental_mat * p2;
        Mat res_up_mat_1 = p1.t() * epipola_line_1;
        double res_up_1 = abs(res_up_mat_1.at<double>(0,0));
        double res_bot_1 = sqrt(pow(epipola_line_1.at<double>(0,0),2) + pow(epipola_line_1.at<double>(1,0),2));
        double distant_1 = res_up_1/res_bot_1;

        Mat epipola_line_2 = p1.t() * fundamental_mat;
        Mat res_up_mat_2 = epipola_line_2 * p2;
        double res_up_2 = abs(res_up_mat_2.at<double>(0,0));
        double res_bot_2 = sqrt(pow(epipola_line_2.at<double>(0,0),2) + pow(epipola_line_2.at<double>(0,1),2));
        double distant_2 = res_up_2/res_bot_2;

        err.push_back(max(distant_1, distant_2));

        if((unsigned int)mask.at<char>(i) != 1){
            total_err_not_de_norm += max(distant_1, distant_2);
            count_err += 1;
            points0.erase(points0.begin() + i);
            points1.erase(points1.begin() + i);
            good_matches.erase(good_matches.begin() +i);

        }
    }
    return fundamental_mat;
}

vector<Point3d> testing(vector<string> imagesPath, Mat K){
    int ceilViewRange = 100;
    vector<Mat> imgs;
    vector<Point3d> Output;
    Ptr<BRISK> detector = BRISK::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::makePtr<cv::FlannBasedMatcher>(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));

    for(int i=0 ;i < imagesPath.size(); i++){
        imgs.push_back(imread( imagesPath[i], IMREAD_GRAYSCALE ));
    }

    //first matching:
    Mat descriptors_1, descriptors_2, descriptors_3;
    vector<KeyPoint> keypoints_0, keypoints_1, keypoints_2;

    vector< DMatch > firstMatch;
    detector->detectAndCompute( imgs[0], Mat(), keypoints_0, descriptors_1 );
    detector->detectAndCompute( imgs[1], Mat(), keypoints_1, descriptors_2 );
    detector->detectAndCompute( imgs[2], Mat(), keypoints_2, descriptors_3 );

    vector< DMatch > match1_2, match2_3;
    match1_2 = matchingTwoViews(keypoints_0, keypoints_1, descriptors_1, descriptors_2 ,matcher);
    match2_3 = matchingTwoViews(keypoints_1, keypoints_2, descriptors_2, descriptors_3 ,matcher);

    vector<Point2d> point_0 = get2DPoints(match1_2, keypoints_0, true);
    vector<Point2d> point_1_0 = get2DPoints(match1_2, keypoints_1, false); // img1-img0
    vector<Point2d> point_1_1 = get2DPoints(match2_3, keypoints_1, true); // img1-img2
    vector<Point2d> point_2 = get2DPoints(match2_3, keypoints_2, false);

    Mat fundamental_mat_1_2 = refineMatching(point_0, point_1_0, match1_2);
    Mat fundamental_mat_2_3 = refineMatching(point_1_1, point_2, match2_3);

    Mat essential_mat_1_2 = K.t()*fundamental_mat_1_2*K;
    Mat essential_mat_2_3 = K.t()*fundamental_mat_2_3*K;

    Mat rotation1_2, translation1_2;
    recoverPose(essential_mat_1_2, point_0, point_1_0, K, rotation1_2, translation1_2);

    Mat rotation2_3, translation2_3;
    recoverPose(essential_mat_2_3, point_1_1, point_2, K, rotation2_3, translation2_3);

    Output = bundleAdjustmentForTwoViews(point_0, point_1_0, rotation1_2, translation1_2, K, point_0.size());
    vector< cv::Point3d > points3D_2;
    points3D_2 = bundleAdjustmentForTwoViews(point_1_1, point_2, rotation2_3, translation2_3, K, point_1_1.size());

    int matchThreeViews = 0;

    for(int i =0; i<point_1_0.size(); i++){
        for(int j =0; j< point_1_1.size(); j++){
            float err = (point_1_0[i].x - point_1_1[j].x) * (point_1_0[i].y - point_1_1[j].y);
            if(abs(err) < 0.0000001){
                matchThreeViews++;
                cout<< Output[i]<< "      "<<points3D_2[j]<<endl;
            }
        }
    }
    cout<<"matchThreeViews     "<<matchThreeViews<<endl;

    Mat M(3, 4, CV_64F);
    M.at<double>(0, 0) = 1.278962600963024;
    M.at<double>(0, 1) = 0.07086204610341218;
    M.at<double>(0, 2) = -0.5684400359896387;
    M.at<double>(0, 3) = 4.553343421856781;

    M.at<double>(1, 0) = 0.0433386151032108;
    M.at<double>(1, 1) = 1.477383293093044;
    M.at<double>(1, 2) = 0.1487245839815894;
    M.at<double>(1, 3) = -1.122386784208388;

    M.at<double>(2, 0) = 0.5838094173305011;
    M.at<double>(2, 1) = -0.05196771079400975;
    M.at<double>(2, 2) = 1.5453825838749;
    M.at<double>(2, 3) = -2.082825078904676;

    vector<Point3d> ps = translateWorld(M,points3D_2);

    Output.insert(Output.end(), ps.begin(), ps.end());
    return Output;
}

