//
// Created by tri on 22/04/2017.
//
#include <GL/glut.h>

using namespace std;
using namespace cv;

void draw(void) {

    // Black background
    glClearColor(0.0f,0.0f,0.0f,1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    //Draw i
    {

    }
    glFlush();

}


int display(int argc, char **argv, std::vector< cv::Point3d > point3ds) {

    glutInit(&argc, argv);

    /*Setting up  The Display
    /    -RGB color model + Alpha Channel = GLUT_RGBA
    */
    glutInitDisplayMode(GLUT_RGBA|GLUT_SINGLE);

    //Configure Window Postion
    glutInitWindowPosition(50, 25);

    //Configure Window Size
    glutInitWindowSize(480,480);

    //Create Window
    glutCreateWindow("Hello OpenGL");


    //Call to the drawing function
//    glutDisplayFunc(draw);

    glClearColor(0.0f,0.0f,0.0f,1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    //Draw i
    {
        for(int i =0; i<point3ds.size(); i++){
            
        }
    }
    glFlush();

    // Loop require by OpenGL
    glutMainLoop();
    return 0;
}
