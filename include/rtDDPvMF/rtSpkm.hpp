/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <signal.h>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <Eigen/Dense>

//#include <pcl/impl/point_types.hpp>

#include <dpMMlowVar/sphericalData.hpp>
#include <dpMMlowVar/kmeansCUDA.hpp>
#include <dpMMlowVar/SO3.hpp>

#include <cudaPcl/dirSeg.hpp>

using namespace Eigen;

//#define RM_NANS_FROM_DEPTH
// normals without nans
//TimerLog: stats over timer cycles (mean +- 3*std):  3.99052+- 6.63986 16.312+- 10.869 43.0436+- 19.7957
// full depth image with nans
//TimerLog: stats over timer cycles (mean +- 3*std):  4.58002+- 9.72736 19.0138+- 17.9823 49.9746+- 30.6944


class RtSpkm : public cudaPcl::DirSeg
{
  public:
    RtSpkm(std::string pathOut, uint32_t K,
      const cudaPcl::CfgSmoothNormals& cfgNormals);
    ~RtSpkm();

    virtual MatrixXf centroids(){return pspkm_->centroids();};

    double residual_;
    uint32_t nIter_;
  protected:
    ofstream fout_;

    uint32_t K_;
    dplv::KMeansCUDA<float,dplv::Spherical<float> >* pspkm_;

    /*
     * runs the actual compute; assumes that normalExtract_ contains
     * the normals on GPU already.
     */
    virtual void compute_();
    /* get lables in input format */
    virtual void getLabels_()
    {
      normalExtract_->uncompressCpu(pspkm_->z().data(),
          pspkm_->z().rows(), z_.data(), z_.rows());
    };
};
// ---------------------------------- impl -----------------------------------


RtSpkm::RtSpkm(std::string pathOut, uint32_t K,
      const cudaPcl::CfgSmoothNormals& cfgNormals)
  : DirSeg(cfgNormals, pathOut),
  residual_(0.0), nIter_(10),
  fout_((pathOut+std::string("./stats.log")).data(),ofstream::out),
  K_(K)
{
  cout<<"inititalizing optSO3"<<endl;
  shared_ptr<MatrixXf> tmp(new MatrixXf(3,1));
  (*tmp) << 1,0,0; // init just to get the dimensions right.
  cld_ = shared_ptr<jsc::ClDataGpuf>(new jsc::ClDataGpuf(tmp,K));
  pspkm_ =  new dplv::KMeansCUDA<float,dplv::Spherical<float> >(cld_);
}

RtSpkm::~RtSpkm()
{
  if(pspkm_) delete pspkm_;
  fout_.close();
}

void RtSpkm::compute_()
{
  // get compressed normals
  tLog_.toctic(1,2);
  int32_t nComp = 0;
  float* d_nComp = normalExtract_->d_normalsComp(nComp);
  cout<<" -- compressed to "<<nComp<<" normals"<<endl;
  pspkm_->nextTimeStepGpu(d_nComp,nComp,3,0,false);
  for(uint32_t i=0; i<nIter_; ++i)
  {
    cout<<"@"<<i<<" :"<<endl;
    pspkm_->updateLabels();
    pspkm_->updateCenters();
    if(pspkm_->convergedCounts(nComp/100)) break;
  }
  pspkm_->getZfromGpu(); // cache z_ back from gpu
  tLog_.toctic(3,4); // total time
  pspkm_->updateState();
  tLog_.toc(4);
  if(tLog_.startLogging()) pspkm_->dumpStats(fout_);
  tLog_.logCycle();
  tLog_.printStats();
  haveLabels_ = false;
}
