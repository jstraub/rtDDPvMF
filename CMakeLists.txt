cmake_minimum_required(VERSION 2.8.1)

include(ExternalProject)
#ExternalProject_add(dpMM
#  GIT_REPOSITORY /data/vision/fisher/code/jstraub/gitrep/research/dpMM.git/
#  GIT_TAG master
#  UPDATE_COMMAND git pull
#  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/dpMM                                   
#  CMAKE_COMMAND cmake .
#  BUILD_COMMAND make -j6 #dpmm #dpStickGMM dpDirMM
#  #INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}
#  INSTALL_COMMAND cmake -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/dpMM -P cmake_install.cmake
#  #INSTALL_COMMAND make install PREFIX=${CMAKE_CURRENT_BINARY_DIR}/dpMM
#)   

ExternalProject_add(dpMMlowVarLib
  GIT_REPOSITORY /data/vision/fisher/code/jstraub/gitrep/research/dpMMlowVar.git/
  GIT_TAG master
  UPDATE_COMMAND git pull
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/dpMMlowVar
  CMAKE_COMMAND cmake .
  BUILD_COMMAND make -j6 #dpmm #dpStickGMM dpDirMM
  #INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}
  INSTALL_COMMAND cmake -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/dpMMlowVar -P cmake_install.cmake
  #INSTALL_COMMAND make install PREFIX=${CMAKE_CURRENT_BINARY_DIR}/dpMM
)   

ExternalProject_add(libCudaPcl                                                     
  GIT_REPOSITORY /data/vision/fisher/code/jstraub/gitrep/research/cudaPcl.git   
  GIT_TAG master
  UPDATE_COMMAND git pull                                                       
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/cudaPcl                                    
  CMAKE_COMMAND cmake .                                                         
  BUILD_COMMAND make -j6 #rtmf #realtimeMF_openni                               
  #INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}                                      
  INSTALL_COMMAND cmake -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/cudaPcl -P cmake_install.cmake
  #INSTALL_COMMAND make install PREFIX=${CMAKE_CURRENT_BINARY_DIR}/rtmf         
  ) 

#ExternalProject_add(libRtMF
#  GIT_REPOSITORY /data/vision/fisher/code/jstraub/gitrep/research/rtmf.git/
#  UPDATE_COMMAND git pull
#  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/rtmf
#  CMAKE_COMMAND cmake .
#  BUILD_COMMAND make -j6 #rtmf #realtimeMF_openni
#  #INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}
#  INSTALL_COMMAND cmake -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/rtmf -P cmake_install.cmake
#  #INSTALL_COMMAND make install PREFIX=${CMAKE_CURRENT_BINARY_DIR}/rtmf
#)   

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
  ./build/dpMMlowVar/include
#  ./build/dpMM/include
  ./build/cudaPcl/include
  ${EIGEN3_INCLUDE_DIR}                                       
  /usr/include                                        
  #$ENV{WORKSPACE_HOME}/3rdparty/pcl-pcl-1.7.1.bak/
    $ENV{WORKSPACE_HOME}/3rdparty/include/pcl-1.7/
    $ENV{WORKSPACE_HOME}/3rdparty/include/vtk-5.8/
  #/usr/include/pcl-1.7/
  /usr/include/ni
  #  /usr/include/vtk-5.8/
  ${Boost_INCLUDE_DIRS}
)                                                                               
                                                                                
link_directories(
  $ENV{WORKSPACE_HOME}/3rdparty/lib/vtk-5.8/
  ${Boost_LIBRARY_DIRS}
  $ENV{WORKSPACE_HOME}/3rdparty/pcl-pcl-1.7.1/build
  ./build/dpMMlowVar/lib/
#  ./build/dpMM/lib/
  ./build/cudaPcl/lib/
)

cuda_add_executable(realtimeDDPvMF_openni
  ./src/realtimeDDPvMF_openni.cpp
  ./src/pcl_helpers.cpp
#  ./src/rtDDPvMF.cpp
#  ./src/realtimeDDPvMF.cpp
  )

target_link_libraries (realtimeDDPvMF_openni
  dpMMlowVar cudaPcl
  boost_system boost_thread boost_program_options
  pcl_common pcl_io pcl_visualization pcl_filters
  OpenNI XnVNite_1_5_2
  vtkCommon vtkFiltering vtkRendering vtkGraphics
  opencv_core opencv_highgui opencv_contrib opencv_imgproc)

ADD_DEPENDENCIES(realtimeDDPvMF_openni dpMM libCudaPcl)

install(DIRECTORY include/ DESTINATION include/                                 
  FILES_MATCHING PATTERN "*.h")                                         
install(DIRECTORY include/ DESTINATION include/                                 
  FILES_MATCHING PATTERN "*.hpp")  
