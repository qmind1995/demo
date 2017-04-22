cmake_minimum_required(VERSION 3.7)
project(DisplayImage)

set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_MODULE_PATH ${CMAKE_INSTALL_PREFIX}/lib/cmake/ )
find_package( OpenCV REQUIRED )
find_package(cvsba REQUIRED )

find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIRS})
link_directories(${GLUT_LIBRARY_DIRS})
add_definitions(${GLUT_DEFINITIONS})

if(NOT GLUT_FOUND)
    message(ERROR " GLUT not found!")
endif(NOT GLUT_FOUND)


find_package(OpenGL REQUIRED)
include_directories(${OpenGL_INCLUDE_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
add_definitions(${OpenGL_DEFINITIONS})
if(NOT OPENGL_FOUND)
    message(ERROR " OPENGL not found!")
endif(NOT OPENGL_FOUND)


set(SOURCE_FILES main.cpp)
add_executable( DisplayImage main.cpp)
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( DisplayImage ${OpenCV_LIBS} ${cvsba_LIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})
