//
// Created by tri on 02/05/2017.
//

#include "ImgsData.h"

ImgsData::ImgsData(vector<string> paths) {
    imgsPath = paths;
}

void ImgsData::restructVisi(vector< vector<int> > _visi, int nview) {

    //from visi -> this.visibility
    //from now, visibility is vector of vector (0 or 1 only!) I LOVE U

    int n3dPoint = _visi[nview-1].size(); // be cau se :last track size is biggest

    for(int v = 0; v <nview; v++){
        for(int i=0; i < n3dPoint; i++){

            if(_visi[v].size() - 1 < i ){
                _visi[v].push_back(0);
            }
            else if(_visi[v][i] == -1 ){
                _visi[v][i] = 0;
            }
            else{
                _visi[v][i] = 1;
            }
        }
    }

    visibility = _visi;

}



vector< DMatch > ImgsData::matching2Views(string imgPath1, string imgPath2 ) {

    Ptr<BRISK> detector = BRISK::create();
    Ptr<DescriptorMatcher> matcher = makePtr<FlannBasedMatcher>(makePtr<cv::flann::LshIndexParams>(12, 20, 2));

    Mat img_1 = imread( imgPath1, IMREAD_GRAYSCALE );
    Mat img_2 = imread( imgPath2, IMREAD_GRAYSCALE );
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
        if(matchesTwoToOne[matchesOneToTwo[i].trainIdx].trainIdx == i && matchesOneToTwo[i].distance <=30){
            matches.push_back(DMatch(i, matchesOneToTwo[i].trainIdx, matchesOneToTwo[i].distance));
        }
    }
    return matches;
}


/**
 * matching n views
 * it return 2d point track and visibility matrix!
 */
void ImgsData::matching() {

    Ptr<BRISK> detector = BRISK::create();
    const int nview  = imgsPath.size();
    vector< vector<int> > _visibility(nview);

    // get img feature:
    vector<Mat> descriptors;
    descriptors.resize(nview);
    vector< vector<KeyPoint> > imgsKeypoints;
    imgsKeypoints.resize(nview);

    for(int i= 0; i < nview; i++){
        vector<KeyPoint> keypoints;
        Mat des;
        Mat img = imread( imgsPath[i], IMREAD_GRAYSCALE );

        detector->detectAndCompute( img, Mat(), keypoints, des );
        imgsKeypoints[i] = keypoints;
        descriptors[i] = des;
    }

    // matching 2 first img
    vector< DMatch > firstMatch = matching2Views(imgsPath[0], imgsPath[1]); // just hot fix
    int firstVR = firstMatch.size();

    // init visibility but, at first, visibility contains match infomation.
    //first view visibility:
    vector<int> visi_0;
    vector<int> visi_1;
    for(int i = 0; i< firstVR; i++){
        visi_0.push_back(1);
        visi_1.push_back(firstMatch[i].trainIdx);
    }
    _visibility[0] = visi_0;
    _visibility[1] = visi_1;

    vector<Point2d> points0(firstVR);
    vector<Point2d> points1(firstVR);

    for(int i = 0; i< firstVR; i++){
        float x0 = imgsKeypoints[0][firstMatch[i].queryIdx].pt.x;
        float y0 = imgsKeypoints[0][firstMatch[i].queryIdx].pt.y;

        float x1 = imgsKeypoints[1][firstMatch[i].trainIdx].pt.x;
        float y1 = imgsKeypoints[1][firstMatch[i].trainIdx].pt.y;

        points0[i] = Point2d(x0, y0);
        points1[i] = Point2d(x1, y1);
    }

    // matching for more view:
    unsigned long prevVR = firstVR;
    int prevVisiIdx = 1;
    for(int view = 1; view < nview -1; view++){
        vector< DMatch > matches = matching2Views(imgsPath[1], imgsPath[2]);
        int vr = matches.size();
        //hot fix, need optimize later:
        int currVisiIdx = prevVisiIdx + 1;
        for(int i = 0; i < prevVR; i++){
            if(_visibility[prevVisiIdx][i] == -1){
                _visibility[currVisiIdx].push_back(-1);
                continue;
            }
            bool found = false;
            for(int j =0; j< vr; j++){
                if(_visibility[prevVisiIdx][i] == matches[j].queryIdx){
                    _visibility[currVisiIdx].push_back(matches[j].trainIdx);
                    matches.erase(matches.begin() + j);
                    vr -- ;
                    found = true;
                    continue;
                }
            }
            if(!found){
                _visibility[currVisiIdx].push_back(-1);
            }
        }

        for(int i=0; i< vr; i++){
            _visibility[prevVisiIdx].push_back(1);
            int clone = matches[i].trainIdx+0;
            _visibility[currVisiIdx].push_back(clone);
        }
        prevVisiIdx++;
    }

    //set attributes
    restructVisi(_visibility, nview);
}
/**
 *
 * @param imgsPath
 * @param firstR
 * @param firstT
 * @param Ks
 * @param showMatching
 * @param ceilViewRange
 * @return
 */
vector<Point3d> get3DPoints(vector<string> imgsPath, Mat &firstR, Mat &firstT , Mat K, bool showMatching, int ceilViewRange = 100){

    int NView = imgsPath.size();
    // Matching


    //get camera pose



    vector<Point3d> points3D;

//    BAForMultiViews()
    return points3D;
}