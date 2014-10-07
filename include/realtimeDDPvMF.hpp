#ifndef REALTIME_MF_HPP_INCLUDED
#define REALTIME_MF_HPP_INCLUDED
#include <root_includes.hpp>
#include <defines.h>

#include <signal.h>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>

#include <Eigen/Dense>

// CUDA runtime
#include <cuda_runtime.h>

// Utilities and system includes
//#include <helper_functions.h>
#include <nvidia/helper_cuda.h>

#include <cuda_pc_helpers.h>
#include <convolutionSeparable_common.h>
#include <convolutionSeparable_common_small.h>
#include <cv_helpers.hpp>
#include <pcl_helpers.hpp>
#include <pcl/impl/point_types.hpp>

#include <vtkWindowToImageFilter.h>
#include <vtkPNGWriter.h>

#include <timer.hpp>
#include <timerLog.hpp>

#include <clusterer.hpp>
#include <ddpvMFmeans.hpp>
#include <ddpvMFmeansCUDA.hpp>
#include <normalExtractCUDA.hpp>

extern void bilateralFilterGPU(float *in, float* out, int w, int h,
      uint32_t radius, float sigma_spatial, float sigma_I);

using namespace Eigen;

class RealtimeDDPvMF
{
  public:
    RealtimeDDPvMF(std::string mode);
    ~RealtimeDDPvMF();

    /* process a depth image of size w*h and return rotation estimate
     * mfRc
     */
    Matrix3f depth_cb(const uint16_t *data, int w,int h);

    void visualizePc();

    void run();

    TimerLog tLog_;
    double residual_;
    uint32_t nIter_;

  protected:

    static const uint32_t SUBSAMPLE_STEP = 1;

    uint32_t nFrame_;
    string resultsPath_;
    float invF_;
    ofstream fout_;
    bool update, updateRGB_;
    boost::mutex updateModelMutex;

    std::string mode_;

    NormalExtractGpu<float> normalExtractor_;

    pcl::PointCloud<pcl::PointXYZRGB> n_;
    pcl::PointCloud<pcl::PointXYZRGB> nDisp_;
    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr n_cp_;

    pcl::PointCloud<pcl::PointXYZ> pc_;
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr pc_cp_;

    uint32_t K_;
    VectorXu z_;
    MatrixXf centroids_;
    cv::Mat rgb_;
    cv::Mat Iz_;

    boost::shared_ptr<MatrixXf> spx_; // normals
    float lambda_, beta_, Q_;
    boost::mt19937 rndGen_;
//    DDPvMFMeans<float>* pddpvmf_;
    DDPvMFMeansCUDA<float>* pddpvmf_;

    virtual void run_impl() = 0;
    virtual void run_cleanup_impl() = 0;


    float JET_r[256];
    float JET_g[256];
    float JET_b[256];

};

#endif
