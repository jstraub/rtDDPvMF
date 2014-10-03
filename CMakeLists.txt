cmake_minimum_required(VERSION 2.8.1)

include(ExternalProject)
ExternalProject_add(dpMM
  GIT_REPOSITORY /data/vision/fisher/code/jstraub/gitrep/research/dpMM.git/
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/dpMM                                   
  CMAKE_COMMAND cmake .
  BUILD_COMMAND make dpmm                                                           
  INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}
  #INSTALL_COMMAND make install PREFIX=${CMAKE_CURRENT_BINARY_DIR}/dpMM
)   

ExternalProject_add(libRtMF
  GIT_REPOSITORY /data/vision/fisher/code/jstraub/gitrep/research/rtmf.git/
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/rtmf
  CMAKE_COMMAND cmake .
  BUILD_COMMAND make rtmf                                           
  INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}
  #INSTALL_COMMAND make install PREFIX=${CMAKE_CURRENT_BINARY_DIR}/rtmf
)   

project(rtDDPvMF)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
  
set(CMAKE_COLOR_MAKEFILE ON)                                                    
IF(NOT CMAKE_BUILD_TYPE)                                                        
  set(CMAKE_BUILD_TYPE Debug) # Release; debug; ...                             
  set(CMAKE_BUILD_TYPE Release) # Release; debug; ...                           
ENDIF()                                                                         

# add flags depending on build type                                             
set(CMAKE_CXX_FLAGS_RELEASE "-g -Wall -Wno-deprecated -O2 -finline-functions -DNDEBUG -fopenmp")         
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wno-deprecated")                                                 
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=\"sm_35,compute_35\")
                                                                                
find_package (Eigen3 REQUIRED)
find_package (CUDA REQUIRED)
find_package (OpenCV REQUIRED)
FIND_PACKAGE (Boost 1.46.1 REQUIRED)

include_directories(                                                            
  include
  ${EIGEN3_INCLUDE_DIR}                                       
  /usr/include                                        
  #$ENV{WORKSPACE_HOME}/3rdparty/pcl-pcl-1.7.1.bak/
    $ENV{WORKSPACE_HOME}/3rdparty/include/pcl-1.7/
    $ENV{WORKSPACE_HOME}/3rdparty/include/vtk-5.8/
  #/usr/include/pcl-1.7/
  /usr/include/ni
  #  /usr/include/vtk-5.8/

  ${Boost_INCLUDE_DIRS}
  $ENV{WORKSPACE_HOME}/research/dpMM/include
  $ENV{WORKSPACE_HOME}/research/rtmf/include
  $ENV{WORKSPACE_HOME}/research/rtmf/common/inc
)                                                                               
                                                                                
link_directories(
  $ENV{WORKSPACE_HOME}/3rdparty/lib/vtk-5.8/
  ${Boost_LIBRARY_DIRS}
  $ENV{WORKSPACE_HOME}/3rdparty/pcl-pcl-1.7.1/build
  $ENV{WORKSPACE_HOME}/research/dpMM/build/
  $ENV{WORKSPACE_HOME}/research/rtmf/build
)

cuda_add_executable(realtimeDDPvMF_openni
  ./src/realtimeDDPvMF_openni.cpp
  ./src/realtimeDDPvMF.cpp
  )

target_link_libraries (realtimeDDPvMF_openni
  dpmm rtmf
  boost_system boost_thread boost_program_options
  pcl_common pcl_io pcl_visualization pcl_filters
  OpenNI XnVNite_1_5_2
  vtkCommon vtkFiltering vtkRendering vtkGraphics
  opencv_core opencv_highgui opencv_contrib)

