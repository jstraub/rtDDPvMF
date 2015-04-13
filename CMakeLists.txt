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

set(LCM OFF)
set(DRC OFF)
set(OPENNI ON)

find_package (Eigen3 REQUIRED)
find_package (CUDA REQUIRED)
FIND_PACKAGE (Boost 1.54 REQUIRED)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
  ${CUDA_TOOLKIT_INCLUDE}
  ${VTK_INCLUDE_DIRS}
)

link_directories(
  ${Boost_LIBRARY_DIRS}
)

if(NOT DRC)
  find_package (OpenCV REQUIRED)
  find_package (PCL REQUIRED)
  message(${OpenCV_LIBS})
  message(${OpenCV_INCLUDE_DIRS})
  message(${PCL_INCLUDE_DIRS})

  include_directories(
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    )

  link_directories(
    ${PCL_LIBRARY_DIRS}
    )
  message( ${OpenCV_INCLUDE_DIRS} )
  message( ${OpenCV_LIBRARIES} )
endif()

message( ${Boost_INCLUDE_DIRS} )
message( ${Boost_LIBRARY_DIRS} )

if(OPENNI)

  cuda_add_executable(realtimeDDPvMF_openni
    ./src/realtimeDDPvMF_openni.cpp
    ./src/pcl_helpers.cpp
    #  ./src/rtDDPvMF.cpp
    #  ./src/realtimeDDPvMF.cpp
    )

  target_link_libraries (realtimeDDPvMF_openni
    jsCore dpMMlowVar cudaPcl
    boost_system boost_thread boost_program_options
    pcl_common pcl_io pcl_visualization pcl_filters pcl_search
    pcl_features
    OpenNI XnVNite_1_5_2
    vtkCommon vtkFiltering vtkRendering vtkGraphics
#    opencv_core opencv_highgui opencv_contrib opencv_imgproc
    ${OpenCV_LIBS}
    )

  pods_install_executables(realtimeDDPvMF_openni)
  pods_install_headers(
    ./include/rtDDPvMF/pcl_helpers.hpp
    ./include/rtDDPvMF/root_includes.hpp
    ./include/rtDDPvMF/rtDDPvMF.hpp
    ./include/rtDDPvMF/rtSpkm.hpp
    ./include/rtDDPvMF/realtimeDDPvMF_openni.hpp
    ./include/rtDDPvMF/rtSpkm_openni.hpp
    DESTINATION rtDDPvMF)

  cuda_add_executable(realtimeDDPvMF_file
    ./src/realtimeDDPvMF_file.cpp
    ./src/pcl_helpers.cpp
    )

  target_link_libraries (realtimeDDPvMF_file
    jsCore dpMMlowVar cudaPcl
    boost_system boost_thread boost_program_options
    pcl_common pcl_io pcl_visualization pcl_filters pcl_search
    pcl_features
    OpenNI XnVNite_1_5_2
    vtkCommon vtkFiltering vtkRendering vtkGraphics
    ${OpenCV_LIBS}
    )

  pods_install_executables(realtimeDDPvMF_openni realtimeDDPvMF_file)
  pods_install_headers(
    ./include/rtDDPvMF/pcl_helpers.hpp
    ./include/rtDDPvMF/realtimeDDPvMF_openni.hpp
    ./include/rtDDPvMF/root_includes.hpp
    ./include/rtDDPvMF/rtDDPvMF.hpp
    ./include/rtDDPvMF/rtSpkm.hpp
    ./include/rtDDPvMF/rtSpkm_openni.hpp
    DESTINATION rtDDPvMF)
endif()

if(LCM OR DRC)

  cuda_add_executable(realtimeDDPvMF_lcm
    ./src/realtimeDDPvMF_lcm.cpp
    ./src/pcl_helpers.cpp
    ./src/jpeg-utils-ijg.c
    )

  target_link_libraries (realtimeDDPvMF_lcm
    jsCore dpMMlowVar cudaPcl
    multisense_image_utils
    boost_system boost_thread boost_program_options
    pcl_common pcl_io pcl_visualization pcl_filters pcl_search
    pcl_features
    OpenNI XnVNite_1_5_2
    vtkCommon vtkFiltering vtkRendering vtkGraphics
    opencv_core opencv_highgui opencv_contrib opencv_imgproc
    opencv_contrib
    opencv_calib3d
    jpeg z png
    )

  pods_use_pkg_config_packages(realtimeMF_lcm
    lcm
    )

  pods_install_executables(realtimeDDPvMF_lcm)
  pods_install_headers(
    ./include/rtDDPvMF/pcl_helpers.hpp
    ./include/rtDDPvMF/root_includes.hpp
    ./include/rtDDPvMF/rtDDPvMF.hpp
    ./include/rtDDPvMF/rtSpkm.hpp
    ./include/rtDDPvMF/realtimeDDPvMF_lcm.hpp
    ./include/rtDDPvMF/jpeg-utils-ijg.h
    DESTINATION rtDDPvMF)

endif()

