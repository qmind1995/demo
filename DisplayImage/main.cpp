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
#include "get3Dpoints.cpp"
#include <fstream>

using namespace std;
using namespace cv;
void readme();
/*
 * @function main
 * @brief Main function
 */

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

int main( int argc, char** argv ){

    Mat K(3, 3, CV_64F);
    //Rhino K
    K.at<double>(0, 0) = 3310.400000;
    K.at<double>(0, 1) = 0.000000;
    K.at<double>(0, 2) = 316.730000;

    K.at<double>(1, 0) = 0.000000;
    K.at<double>(1, 1) = 3325.500000;
    K.at<double>(1, 2) = 200.550000;

    K.at<double>(2, 0) = 0.000000;
    K.at<double>(2, 1) = 0.000000;
    K.at<double>(2, 2) = 1.000000;

    //Temple K:
//    K.at<double>(0, 0) = 1520.400000;
//    K.at<double>(0, 1) = 0.000000;
//    K.at<double>(0, 2) = 302.320000;
//
//    K.at<double>(1, 0) = 0.000000;
//    K.at<double>(1, 1) = 1525.900000;
//    K.at<double>(1, 2) = 246.870000;
//
//    K.at<double>(2, 0) = 0.000000;
//    K.at<double>(2, 1) = 0.000000;
//    K.at<double>(2, 2) = 1.000000;
    bool showMatching = true;

    int numImageTest = 2;

    string imageFolder = "./pictures/dinoRing/";
    string imageListFile = "./pictures/dinoRing/dinoR_good_silhouette_images.txt";
//    string imageFolder = "./templeRing/";
//    string imageListFile = "./templeRing/listImg.txt";
    vector<string> imageList;
    imageList.resize(numImageTest);
    ifstream imgListFileStream (imageListFile);

    if(! imgListFileStream.is_open()){
        return -1;
    }

    for(int i=0; i<numImageTest; i++){
        imgListFileStream >>imageList[i];
    }

    vector< cv::Point3d > points3D;
    Mat firstRotation, firstTranslation;
    for(int i =0 ; i< numImageTest-1; i++){
        int image0_index = i ;
        int image1_index = i + 1 ;
        cout << image0_index << endl;
        string image0 = imageFolder + imageList[image0_index];
        string image1 = imageFolder + imageList[image1_index];




        get3DPointsFromTwoImg(points3D,image0, image1, K, showMatching, 100);

    }

    writeMeshLabFile("test.ply",points3D);

    waitKey(0);
    return 0;
}
