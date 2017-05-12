#include "get3Dpoints.cpp"

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

    int numImageTest = 3;

    string imageFolder = "./pictures/ImageDataset_SceauxCastle-master/images/";
    string imageListFile = "./pictures/ImageDataset_SceauxCastle-master/images/list_name.txt";
    vector<string> imageList;
    imageList.resize(numImageTest);
    ifstream imgListFileStream (imageListFile);

    if(! imgListFileStream.is_open()){
        return -1;
    }

//    for(int i=0; i<numImageTest; i++){
//        imgListFileStream >>imageList[i];
//    }
    imageList[0]= "./pictures/ImageDataset_SceauxCastle-master/images/100_7100.JPG";
    imageList[1]= "./pictures/ImageDataset_SceauxCastle-master/images/100_7101.JPG";
    imageList[2]= "./pictures/ImageDataset_SceauxCastle-master/images/100_7102.JPG";

    vector< cv::Point3d > points3D;
    Mat firstRotation, firstTranslation;

    firstRotation = cv::Mat::eye(3, 3, CV_64F);
    firstTranslation = cv::Mat::zeros(3, 1, CV_64F);
    int readImgIndex = 0;
    int maxMatchImg  = 4;

    //for test
    vector<string> imgsPath;
    imgsPath.push_back(imageList[0]);
    imgsPath.push_back(imageList[1]);
    imgsPath.push_back(imageList[2]);
    ImgsData  data(imgsPath);

    data.matching();
    //for test
    /*
    while(readImgIndex < numImageTest){
        int image0_index = readImgIndex ;
        int image1_index = readImgIndex + 1 ;
        readImgIndex = image1_index;

        bool isGet3DpointsSuccess = false;
        vector<string> imgsPath;
        imgsPath.push_back(imageList[image0_index]);
        imgsPath.push_back(imageList[image1_index]);

        ImgsData  data(imgsPath);

        data.matching();

        while(!isGet3DpointsSuccess && imgsPath.size() <=  maxMatchImg){
            vector<Point3d> solvedPoints = data.get3DPoints();
            if(solvedPoints.size() > 0 ){
                points3D.insert(points3D.end(), solvedPoints.begin(), solvedPoints.end());
                isGet3DpointsSuccess  = true;
            }
            else{
                readImgIndex++;
                string moreImg = imageList[readImgIndex];
                imgsPath.push_back(moreImg);
            }
        }
    }
    */
    writeMeshLabFile("test.ply",points3D);

    waitKey(0);
    return 0;
}
