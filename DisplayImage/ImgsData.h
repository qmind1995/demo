//
// Created by tri on 11/05/2017.
//
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
#include <fstream>

using namespace std;
using namespace cv;

class ImgsData{
private:
    vector<Point3d> points3D;
    vector<string> imgsPath;
    vector< vector<Point2d> > imgPoints;
    vector< vector<int> > visibility;
    vector<Mat> Rs;
    vector<Mat> Ts;
public:
    ImgsData(vector<string> paths);
    void restructVisi(vector< vector<int> > _visi, int nview);
    void setImgPoints();
    vector< DMatch > matching2Views(string imgPath1, string imgPath2 );
    vector<Point3d> get3DPoints();
    void matching();
};