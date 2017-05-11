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

    bool showMatching = true;

    int numImageTest = 2;

    string imageFolder = "./pictures/dinoRing/";
    string imageListFile = "./pictures/dinoRing/dinoR_good_silhouette_images.txt";
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

    firstRotation = cv::Mat::eye(3, 3, CV_64F);
    firstTranslation = cv::Mat::zeros(3, 1, CV_64F);
    int readImgIndex = 0;
    int maxMatchImg  = 5;

    while(readImgIndex < numImageTest){
        int image0_index = readImgIndex ;
        int image1_index = readImgIndex + 1 ;
        readImgIndex = image1_index;

        string image0 = imageFolder + imageList[image0_index];
        string image1 = imageFolder + imageList[image1_index];

        bool isGet3DpointsSuccess = false;
        vector<string> imgsPath;
        imgsPath.push_back(image0);
        imgsPath.push_back(image1);

        while(!isGet3DpointsSuccess && imgsPath.size() <=  maxMatchImg){
            vector<Point3d> solvedPoints = get3DPoints(imgsPath, firstRotation, firstTranslation, K, showMatching);
            if(solvedPoints.size() > 0 ){
                points3D.insert(points3D.end(), solvedPoints.begin(), solvedPoints.end());
                isGet3DpointsSuccess  = true;
            }
            else{
                readImgIndex++;
                string moreImg = imageFolder + imageList[readImgIndex];
                imgsPath.push_back(moreImg);
            }
        }
    }
    writeMeshLabFile("test.ply",points3D);

    waitKey(0);
    return 0;
}
