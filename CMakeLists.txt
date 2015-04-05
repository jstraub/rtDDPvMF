# Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
# under the MIT license. See the license file LICENSE.

cmake_minimum_required(VERSION 2.8.1)
project(rtDDPvMF)
set(POD_NAME rtDDPvMF)
include(cmake/pods.cmake)

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
FIND_PACKAGE (Boost 1.54 REQUIRED)
FIND_PACKAGE (PCL REQUIRED)


include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
)

link_directories(
  ${PCL_LIBRARY_DIRS}
  ${Boost_LIBRARY_DIRS}
)

cuda_add_executable(realtimeDDPvMF_openni
  ./src/realtimeDDPvMF_openni.cpp
  ./src/pcl_helpers.cpp
#  ./src/rtDDPvMF.cpp
#  ./src/realtimeDDPvMF.cpp
  )

target_link_libraries (realtimeDDPvMF_openni
  jsCore dpMMlowVar cudaPcl
  boost_system boost_thread boost_program_options
  pcl_common pcl_io pcl_visualization pcl_filters
  OpenNI XnVNite_1_5_2
  vtkCommon vtkFiltering vtkRendering vtkGraphics
  opencv_core opencv_highgui opencv_contrib opencv_imgproc)

ADD_DEPENDENCIES(realtimeDDPvMF_openni dpMMlowVarLib libCudaPcl)

#install(DIRECTORY include/rtDDPvMF/ DESTINATION include/rtDDPvMF/
#  FILES_MATCHING PATTERN "*.h")
#install(DIRECTORY include/rtDDPvMF/ DESTINATION include/rtDDPvMF/
#  FILES_MATCHING PATTERN "*.hpp")

pods_install_executables(realtimeDDPvMF_openni)
pods_install_headers(
  ./include/rtDDPvMF/pcl_helpers.hpp
  ./include/rtDDPvMF/realtimeDDPvMF_openni.hpp
  ./include/rtDDPvMF/root_includes.hpp
  ./include/rtDDPvMF/rtDDPvMF.hpp
  ./include/rtDDPvMF/rtSpkm.hpp
  ./include/rtDDPvMF/rtSpkm_openni.hpp
  DESTINATION rtDDPvMF)
