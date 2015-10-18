/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <rtDDPvMF/root_includes.hpp>

#include <signal.h>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>


#include <Eigen/Dense>

//#include <pcl/impl/point_types.hpp>

//#include <vtkWindowToImageFilter.h>
//#include <vtkPNGWriter.h>

#include <dpMMlowVar/sphericalData.hpp>
#include <dpMMlowVar/ddpmeansCUDA.hpp>
#include <dpMMlowVar/kmeansCUDA.hpp>
#include <dpMMlowVar/SO3.hpp>

#include <cudaPcl/dirSeg.hpp>

using namespace Eigen;

//#define RM_NANS_FROM_DEPTH
// normals without nans
//TimerLog: stats over timer cycles (mean +- 3*std):  3.99052+- 6.63986 16.312+- 10.869 43.0436+- 19.7957
// full depth image with nans
//TimerLog: stats over timer cycles (mean +- 3*std):  4.58002+- 9.72736 19.0138+- 17.9823 49.9746+- 30.6944

void rotatePcGPU(float* d_pc, float* d_R, int32_t N, int32_t step);

struct CfgRtDDPvMF
{

  CfgRtDDPvMF() : f_d(540.0), lambda(-1.), beta(1e5), Q(-2.),
      nSkipFramesSave(30), nFramesSurvive_(0),lambdaDeg_(90.)
  {};
  CfgRtDDPvMF(const CfgRtDDPvMF& cfg)
    : f_d(cfg.f_d), lambda(cfg.lambda), beta(cfg.beta), Q(cfg.Q),
      nSkipFramesSave(cfg.nSkipFramesSave), nFramesSurvive_(cfg.nFramesSurvive_),
      lambdaDeg_(cfg.lambdaDeg_)
  {
    pathOut = cfg.pathOut;
  };

  double f_d;
  double lambda;
  double beta;
  double Q;

  int32_t nSkipFramesSave;
  std::string pathOut;

  int32_t nFramesSurvive_;
  double lambdaDeg_;

  void lambdaFromDeg(double lambdaDeg)
  {
    lambdaDeg_ = lambdaDeg;
    lambda = cos(lambdaDeg*M_PI/180.0)-1.;
  };
  void QfromFrames2Survive(int32_t nFramesSurvive)
  {
    nFramesSurvive_ = nFramesSurvive;
    Q = nFramesSurvive == 0? -2. : lambda/double(nFramesSurvive);
  };
};

class RtDDPvMF : public cudaPcl::DirSeg
{
  public:
    RtDDPvMF(const CfgRtDDPvMF& cfg,
      const cudaPcl::CfgSmoothNormals& cfgNormals);
    ~RtDDPvMF();

    Matrix3f applyConstVelModel();

    virtual MatrixXf centroids(){return pddpvmf_->centroids();};
    virtual const VectorXu& labels();
    uint32_t GetK() {
      return (pddpvmf_->counts().array() > 0).matrix().cast<uint32_t>().sum();
    }
    MatrixXf GetxSums() { return cld_->xSums();};
//    VectorXf GetCounts() { return cld_->counts();};
    VectorXf GetCounts() {
      uint32_t K = (pddpvmf_->counts().array() > 0).matrix().cast<uint32_t>().sum();
      VectorXf counts(K);
      uint32_t k = 0;
      for (uint32_t i =0; i<pddpvmf_->counts().rows(); ++i)
        if (pddpvmf_->counts()(i) > 0)
          counts(k++) = pddpvmf_->counts()(i);
      return counts;
    };

    MatrixXf GetCentroids() const {
      std::cout << "cs: " << cld_->counts().transpose() << std::endl;
      std::cout << pddpvmf_->centroids() << std::endl;
      uint32_t K = (pddpvmf_->counts().array() > 0).matrix().cast<uint32_t>().sum();
      std::cout << "K: " << K << std::endl;
      MatrixXf centroids(3,K);
      uint32_t k = 0;
      for (uint32_t i =0; i<pddpvmf_->counts().rows(); ++i)
        if (pddpvmf_->counts()(i) > 0)
          centroids.col(k++) = pddpvmf_->centroids().col(i);
      return centroids;
    }

    double residual_;
    uint32_t nIter_;
  protected:

    ofstream fout_;

    CfgRtDDPvMF cfg_;
    dplv::DDPMeansCUDA<float,dplv::Spherical<float> >* pddpvmf_;

    /*
     * runs the actual compute; assumes that normalExtract_ contains
     * the normals on GPU already.
     */
    virtual void compute_();
    /* get lables in input format */
    virtual void getLabels_()
    {
      normalExtract_->uncompressCpu(pddpvmf_->z().data(),
          pddpvmf_->z().rows(), z_.data(), z_.rows());
    };
};
// ---------------------------------- impl -----------------------------------


RtDDPvMF::RtDDPvMF(const CfgRtDDPvMF& cfg,
      const cudaPcl::CfgSmoothNormals& cfgNormals)
  : DirSeg(cfgNormals, cfg.pathOut),
  residual_(0.0), nIter_(10),
  fout_((cfg.pathOut+std::string("./stats.log")).data(),ofstream::out),
  cfg_(cfg)
{
  shared_ptr<MatrixXf> tmp(new MatrixXf(3,1));
  (*tmp) << 1,0,0; // init just to get the dimensions right.
  cld_ = shared_ptr<jsc::ClDataGpuf>(new jsc::ClDataGpuf(tmp,0));
  pddpvmf_ =  new dplv::DDPMeansCUDA<float,dplv::Spherical<float> >
    (cld_, cfg_.lambda, cfg_.Q, cfg_.beta);
}

RtDDPvMF::~RtDDPvMF()
{
  delete pddpvmf_;
  fout_.close();
}

void RtDDPvMF::compute_()
{
  // get compressed normals
  tLog_.toctic(1,2);
  int32_t nComp = 0;
  float* d_nComp = normalExtract_->d_normalsComp(nComp);
  cout<<" -- compressed to "<<nComp<<" normals"<<endl;
  tLog_.toctic(2,3); // total time
  pddpvmf_->nextTimeStepGpu(d_nComp,nComp,3,0,true);//false);
  for(uint32_t i=0; i<nIter_; ++i)
  {
    cout<<"@"<<i<<" :"<<endl;
    pddpvmf_->updateLabels();
    pddpvmf_->updateCenters();
    if(pddpvmf_->convergedCounts(nComp/100)) break;
  }
  pddpvmf_->getZfromGpu(); // cache z_ back from gpu
  tLog_.toctic(3,4); // total time
  pddpvmf_->updateState();
  tLog_.toc(4);
  if(tLog_.startLogging()) pddpvmf_->dumpStats(fout_);
  tLog_.logCycle();
  tLog_.printStats();
  haveLabels_ = false;
}


Matrix3f RtDDPvMF::applyConstVelModel()
{
  Matrix3f deltaR = Matrix3f::Identity();
// get them from internal since they keep track of removed clusters
  MatrixXf centroids = pddpvmf_->centroids();
  MatrixXf prevCentroids =  pddpvmf_->prevCentroids();

  // compute all rotations from different axes
  std::vector<MatrixXf> Rs(std::min(centroids.cols(),
        prevCentroids.cols()));
  for(uint32_t k=0; k<centroids.cols(); ++k)
    if(k < prevCentroids.cols())
    {
      Rs[k] = dplv::rotationFromAtoB<float>(prevCentroids.col(k),
          centroids.col(k));
    }
  // compute the Karcher mean rotation
  if(Rs.size()>0)
  {
    deltaR = dplv::SO3<float>::meanRotation(Rs,
        pddpvmf_->counts().cast<float>(),20);
    pddpvmf_->rotateUninstantiated(deltaR.transpose());
    cout<<"const velocity model applied rotation"<<endl<<deltaR<<endl;
  }
  return deltaR;
}

const VectorXu& RtDDPvMF::labels()
{
  if(!haveLabels_)
    K_ = pddpvmf_->getK();
  DirSeg::labels();
  return z_;
};


